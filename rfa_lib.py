"""rfa_lib: entrenamiento de DenseNet121 sobre NIH ChestX-ray14, refactorizado.

Flujo de uso (en el notebook):

    from rfa_lib import *

    datos = preparar_datos(ds)                                  # una vez: particiones por paciente
    base = construir_config(capas='completo', aug='light')     # todos los parámetros en un diccionario

    # 1. búsqueda de hiperparámetros -> devuelve la mejor configuración completa
    mejores = buscar_hiperparametros(datos, base, {'lr': [1e-4, 3e-4, 1e-3], 'loss': ['bce', 'pw_sqrt']})

    # 2. entrenamiento serio con esa configuración -> devuelve el modelo entrenado
    modelo = entrenar(datos, mejores, nombre='dn_completo')

    # 3. resultados en test (se evalúan UNA vez y se guardan junto al modelo)
    resultados_test(modelo, datos)

    # 4. comparar modelos o promedios de modelos, con intervalos de confianza por pacientes
    comparar(datos, {'completo': modelo, 'completo + otro': [modelo, otro_modelo]})

Todo lo que se entrena o se busca se guarda en la carpeta `modelos/`; si ya existe, se carga en vez de repetirlo.

Convención: lo que NO empieza por «_» es la API pública; lo que empieza por «_» es interno.
`from rfa_lib import *` solo importa lo público (ver __all__).

Cómo se mide: mAUC = media del AUROC de las 14 patologías (sin «No Finding»). La búsqueda y el entrenamiento
usan SOLO validación; el test solo lo toca `resultados_test`.
"""
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import timm
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from timm.scheduler.cosine_lr import CosineLRScheduler
from sklearn.metrics import roc_auc_score
from prettytable import PrettyTable, TableStyle

__all__ = ['CONFIG_DEFECTO', 'ETIQUETAS', 'Datos', 'ModeloEntrenado',
           'preparar_datos', 'construir_config', 'buscar_hiperparametros', 'entrenar', 'cargar_modelo',
           'resultados_test', 'probabilidades', 'etiquetas', 'auc_por_clase', 'mauc', 'promediar',
           'comparar', 'comparar_no_finding', 'graficar_resultados', 'graficar_curvas']

# =====================================================================================================
# Constantes
# =====================================================================================================
ETIQUETAS = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
             'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
             'Pleural Thickening', 'Hernia']
_N_CLASES = len(ETIQUETAS)                                   # 15: «No Finding» + 14 patologías
_N_PATOLOGIAS = _N_CLASES - 1                                # 14
_DISP = 'cuda' if torch.cuda.is_available() else 'cpu'
_MEDIA, _STD = 0.449, 0.226                                  # normalización de ImageNet en escala de grises
_TAM_IMG = 224                                               # las imágenes del dataset son 224x224
_DIR_MODELOS = 'modelos'                                     # aquí se guardan pesos, resultados y búsquedas

# Todos los parámetros con su valor por defecto. `construir_config` parte de aquí.
CONFIG_DEFECTO = {
    # --- qué se entrena ---
    'capas': 'completo',        # 'cabecera' | 'completo' | N (0-5: cabecera + N etapas desde arriba) | 'b2_b3' | 'b3_b4' | lista de prefijos
    'no_finding': 'clase',      # 'clase' (15 salidas, «No Finding» es una clase) | 'derivado' (14 salidas, se deduce)
    'init': None,               # empezar desde un modelo ya entrenado (LP-FT): su nombre, p. ej. 'head', o una ruta .pt
    # --- optimización ---
    'lr': 3e-4,
    'wd': 1e-2,                 # weight decay
    'layer_decay': 1.0,         # <1: las capas más cercanas a la entrada usan un lr menor (lr * layer_decay**etapa)
    'loss': 'pw_sqrt',          # 'bce' | 'pw' (peso neg/pos por clase) | 'pw_sqrt' (su raíz)
    'bs': 32,
    # --- datos ---
    'aug': 'light',             # 'none' | 'light' | 'medium' | 'strong' | 'strong_flip'
    'workers': 6,
    # --- duración y planificador del lr ---
    'epocas': 20,               # máximo de épocas
    'scheduler': 'plateau',     # 'plateau' (baja el lr cuando la validación se estanca) | 'coseno' (calendario fijo)
    'paciencia': 5,             # parada temprana: épocas sin mejorar (None = no parar)
    'lr_factor': 0.3,           # solo 'plateau': el lr se multiplica por esto al estancarse
    'lr_paciencia': 1,          # solo 'plateau': épocas sin mejorar antes de bajar el lr
    # --- otros ---
    'bn_congelada_en_eval': False,   # True: las BatchNorm de capas congeladas no actualizan sus estadísticas
    'semilla': 42,
}
# Aumento de datos que se añade a las transformaciones de TRAIN (nunca a validación ni test). Se elige con cfg['aug'].
_TRANSFORMACIONES_EXTRA = {
        'none':        [],
        'light':       [v2.RandomRotation(5), v2.RandomAffine(0, translate=(.03, .03))],
        'medium':      [v2.RandomRotation(10), v2.RandomAffine(0, translate=(.05, .05), scale=(.95, 1.05)),
                        v2.ColorJitter(brightness=.2, contrast=.2)],
        'strong':      [v2.RandomResizedCrop(_TAM_IMG, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
                        v2.RandomRotation(10), v2.RandomAffine(0, translate=(.05, .05)),
                        v2.ColorJitter(brightness=.3, contrast=.3)],
        'strong_flip': [v2.RandomResizedCrop(_TAM_IMG, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
                        v2.RandomRotation(10), v2.RandomAffine(0, translate=(.05, .05)),
                        v2.ColorJitter(brightness=.3, contrast=.3),
                        v2.RandomHorizontalFlip()],
}
_OPCIONES = {
    'aug': tuple(_TRANSFORMACIONES_EXTRA),
    'loss': ('bce', 'pw', 'pw_sqrt'),
    'scheduler': ('plateau', 'coseno'),
    'no_finding': ('clase', 'derivado'),
}

# Etapas de DenseNet121 ordenadas desde la cabeza hacia la entrada. «capas=N» entrena las etapas 0..N.
_ETAPAS = [
    ('classifier',),
    ('features.norm5', 'features.denseblock4'),
    ('features.transition3', 'features.denseblock3'),
    ('features.transition2', 'features.denseblock2'),
    ('features.transition1', 'features.denseblock1'),
    ('features.conv0', 'features.norm0'),
]
_PRESETS_CAPAS = {
    'cabecera': 0,
    'completo': len(_ETAPAS) - 1,
    # bloques intermedios de los primeros experimentos (se mantienen para poder cargar esos modelos)
    'b2_b3': ('classifier', 'features.denseblock2', 'features.transition2', 'features.denseblock3', 'features.transition3'),
    'b3_b4': ('classifier', 'features.denseblock3', 'features.transition3', 'features.denseblock4', 'features.norm5'),
}


# =====================================================================================================
# Configuración
# =====================================================================================================
def construir_config(base=None, **cambios):
    """Diccionario de configuración COMPLETO y validado: defaults < `base` < `cambios`.
    Valida que no haya claves mal escritas ni valores inexistentes, así un error aparece antes de lanzar nada.
        construir_config(capas=1, lr=1e-4)           # desde los defaults
        construir_config(mejores, epocas=30)          # partiendo de otra configuración
    """
    cfg = {**CONFIG_DEFECTO, **(base or {}), **cambios}
    raras = set(cfg) - set(CONFIG_DEFECTO)
    if raras:
        raise KeyError(f'Parámetros desconocidos: {sorted(raras)}. Válidos: {sorted(CONFIG_DEFECTO)}')
    for clave, validos in _OPCIONES.items():
        if cfg[clave] not in validos:
            raise ValueError(f"'{clave}' = {cfg[clave]!r} no es válido. Opciones: {validos}")
    _prefijos(cfg['capas'])
    return cfg


def _prefijos(capas):
    """Prefijos de nombre de parámetro que se entrenan para un valor de `capas`."""
    if isinstance(capas, str):
        if capas not in _PRESETS_CAPAS:
            raise ValueError(f"capas = {capas!r} no es válido. Opciones: {sorted(_PRESETS_CAPAS)}, un entero 0-5 o una lista de prefijos")
        capas = _PRESETS_CAPAS[capas]
    if isinstance(capas, int):
        if not 0 <= capas < len(_ETAPAS):
            raise ValueError(f'capas (entero) debe estar entre 0 y {len(_ETAPAS) - 1}')
        return tuple(p for etapa in _ETAPAS[:capas + 1] for p in etapa)
    return tuple(capas)


def _etapa(nombre_parametro):
    """Etapa (0 = cabeza ... 5 = entrada) a la que pertenece un parámetro."""
    for i, etapa in enumerate(_ETAPAS):
        if nombre_parametro.startswith(etapa):
            return i
    raise ValueError(f'Parámetro sin etapa: {nombre_parametro}')


def _nombre_auto(cfg):
    capas = cfg['capas'] if isinstance(cfg['capas'], (str, int)) else 'custom'
    return f"dn_{capas if isinstance(capas, str) else f'e{capas}'}" + ('_nf' if cfg['no_finding'] == 'derivado' else '')


# =====================================================================================================
# Datos
# =====================================================================================================
@dataclass
class Datos:
    """Dataset y particiones. Se crea una vez con `preparar_datos` y se pasa a las funciones.
    train/val/test son posiciones (índices) en `ds`; Y son las etiquetas multi-hot de TODO el dataset (N, 15)."""
    ds: object
    Y: torch.Tensor
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    paciente: np.ndarray
    edad: np.ndarray
    sexo: np.ndarray


def preparar_datos(ds, semilla=42, fracciones=(0.70, 0.15), mostrar=True):
    """Partición POR PACIENTE (70/15/15): todas las radiografías de un paciente van a la misma partición, para
    que ninguno aparezca a la vez en train y en validación/test (fuga de información). Calcula también la matriz
    de etiquetas una sola vez, en vez de construirla imagen a imagen durante el entrenamiento."""
    pacientes = np.asarray(ds['patient_id'])
    unicos = np.unique(pacientes)
    np.random.default_rng(semilla).shuffle(unicos)
    posicion = {p: i for i, p in enumerate(unicos)}
    orden = np.fromiter((posicion[p] for p in pacientes), dtype=np.int64, count=len(pacientes))
    n = len(unicos)
    grupo = np.digitize(orden, [int(fracciones[0] * n), int((fracciones[0] + fracciones[1]) * n)])

    Y = torch.tensor([[1. if i in etiquetas else 0. for i in range(_N_CLASES)] for etiquetas in ds['labels']])
    datos = Datos(ds=ds, Y=Y, train=np.where(grupo == 0)[0], val=np.where(grupo == 1)[0], test=np.where(grupo == 2)[0],
                  paciente=pacientes, edad=np.asarray(ds['age']), sexo=np.asarray(ds['sex']))
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        assert not set(pacientes[getattr(datos, a)]) & set(pacientes[getattr(datos, b)]), f'paciente en {a} y {b}'
    if mostrar:
        t = PrettyTable(['', 'Entrenamiento', 'Validación', 'Test', 'Total'])
        t.set_style(TableStyle.SINGLE_BORDER)
        t.add_row(['Imágenes', len(datos.train), len(datos.val), len(datos.test), len(ds)])
        t.add_row(['Pacientes'] + [len(set(pacientes[x])) for x in (datos.train, datos.val, datos.test)] + [n])
        print(t)
    return datos



def _transformaciones(aug):
    """Transformación de las imágenes: base (gris, tensor, normalización) más el aumento `_TRANSFORMACIONES_EXTRA[aug]`.
    Para train se usa `aug` de la configuración; para evaluar (validación, test), `_transformaciones('none')`,
    es decir, la imagen sin perturbar."""
    base = [v2.Grayscale(1), v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
    norm = [v2.Normalize([_MEDIA], [_STD])]
    return v2.Compose(base + _TRANSFORMACIONES_EXTRA[aug] + norm)


class _DatasetRX(Dataset):
    """Una radiografía transformada y su vector de etiquetas. `idx` son posiciones en `ds`."""
    def __init__(self, ds, idx, Y, transformacion):
        self.ds, self.idx, self.Y, self.tf = ds, idx, Y, transformacion

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        return self.tf(self.ds[j]['image']), self.Y[j].clone()


def _loader(datos, idx, cfg, transformacion, barajar):
    Y = datos.Y if cfg['no_finding'] == 'clase' else datos.Y[:, 1:].contiguous()
    w = cfg['workers']
    return DataLoader(_DatasetRX(datos.ds, idx, Y, transformacion), batch_size=cfg['bs'], shuffle=barajar,
                      num_workers=w, pin_memory=_DISP == 'cuda', multiprocessing_context='fork' if w else None)


def _pos_weight(datos, idx, cfg):
    """Peso de los positivos de cada clase en la loss (solo con train). Compensa el desbalanceo:
    una clase rara tiene muchos más negativos que positivos y sin peso el modelo la ignora."""
    if cfg['loss'] == 'bce':
        return None
    pos = datos.Y[torch.as_tensor(idx)].sum(0)
    peso = (len(idx) - pos) / pos.clamp(min=1)
    if cfg['loss'] == 'pw_sqrt':
        peso = peso.sqrt()
    return peso if cfg['no_finding'] == 'clase' else peso[1:]


# =====================================================================================================
# Modelo
# =====================================================================================================
def _crear_modelo(cfg, preentrenado=True):
    """DenseNet121 de 1 canal con la cabeza adecuada y los parámetros entrenables que marque `cfg['capas']`.
    El modelo se mueve ya a la GPU (el bucle de entrenamiento no tiene que ocuparse de ello)."""
    salidas = _N_CLASES if cfg['no_finding'] == 'clase' else _N_PATOLOGIAS
    modelo = timm.create_model('densenet121.tv_in1k', pretrained=preentrenado, num_classes=salidas, in_chans=1)
    if cfg['init']:
        ruta = cfg['init'] if cfg['init'].endswith('.pt') else _ruta(cfg['init'])
        modelo.load_state_dict(torch.load(ruta, map_location='cpu', weights_only=True))
    prefijos = _prefijos(cfg['capas'])
    for nombre, p in modelo.named_parameters():
        p.requires_grad = nombre.startswith(prefijos)
    return modelo.to(_DISP).to(memory_format=torch.channels_last)


def _grupos(modelo, lr, layer_decay):
    """Un grupo de parámetros por etapa; con layer_decay<1 las etapas cercanas a la entrada usan un lr menor."""
    grupos = []
    for e in range(len(_ETAPAS)):
        ps = [p for n, p in modelo.named_parameters() if p.requires_grad and _etapa(n) == e]
        if ps:
            grupos.append({'params': ps, 'lr': lr * layer_decay ** e, 'base_lr': lr * layer_decay ** e})
    return grupos


def _modo_train(modelo, cfg):
    modelo.train()
    if cfg['bn_congelada_en_eval']:
        for m in modelo.modules():
            if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
                m.eval()


def _semilla(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# =====================================================================================================
# Métricas
# =====================================================================================================
def _auc_por_clase(P, Y):
    """AUROC de cada columna (sklearn, que trata bien los empates). nan si no hay positivos o negativos."""
    P, Y = np.asarray(P), np.asarray(Y)
    return torch.tensor([roc_auc_score(Y[:, c], P[:, c]) if 0 < Y[:, c].sum() < len(Y) else float('nan')
                         for c in range(P.shape[1])])


def _mauc(aucs):
    """Media de las 14 patologías (excluye «No Finding», que es la primera salida cuando hay 15)."""
    return aucs[-_N_PATOLOGIAS:].nanmean().item()


def _amp():
    return torch.autocast(device_type=_DISP, dtype=torch.float16, enabled=_DISP == 'cuda')


@torch.no_grad()
def _predecir(modelo, loader):
    """Probabilidades (sigmoide) y etiquetas de todo el loader."""
    modelo.eval()
    P, Y = [], []
    for X, y in loader:
        with _amp():
            salida = modelo(X.to(_DISP).contiguous(memory_format=torch.channels_last))
        P.append(torch.sigmoid(salida.float()).cpu())
        Y.append(y)
    return torch.cat(P), torch.cat(Y)


def _puntuaciones_nf(P):
    """Puntuación de «No Finding» (alta = no hay patología) a partir de las 14 probabilidades:
    'prod' = prod(1 - p) (prob. de que no haya ninguna, suponiendo independencia); 'max' = 1 - max(p)."""
    return {'prod': (1 - P).prod(1), 'max': 1 - P.max(1).values}


# =====================================================================================================
# Entrenamiento
# =====================================================================================================
class _Planificador:
    """Decide el learning rate en cada momento. Dos modos, elegidos con cfg['scheduler']:
      'plateau'  calentamiento lineal durante la 1ª época; después el lr se multiplica por `lr_factor` cuando
                 el mAUC de validación no mejora durante `lr_paciencia` épocas.
      'coseno'   calentamiento de 1 época y bajada en coseno hasta lr/100 al final de `epocas` (timm, TL08)."""
    def __init__(self, opt, cfg):
        self.opt, self.tipo = opt, cfg['scheduler']
        if self.tipo == 'coseno':
            self.s = CosineLRScheduler(opt, t_initial=cfg['epocas'], lr_min=cfg['lr'] / 100,
                                       warmup_t=1, warmup_lr_init=cfg['lr'] / 100)
        else:
            self.s = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='max', factor=cfg['lr_factor'], patience=cfg['lr_paciencia'], min_lr=1e-6)

    def inicio_epoca(self, ep):
        if self.tipo == 'coseno':
            self.s.step(ep)

    def iteracion(self, ep, it, n_it):
        if self.tipo == 'plateau' and ep == 0:
            f = 0.01 + 0.99 * it / max(n_it - 1, 1)
            for g in self.opt.param_groups:
                g['lr'] = g['base_lr'] * f

    def fin_epoca(self, val):
        if self.tipo == 'plateau':
            self.s.step(val)

    def lr(self):
        return max(g['lr'] for g in self.opt.param_groups)


def _bucle(modelo, tr, va, cfg, pos_weight, ruta, verbose):
    """El bucle de entrenamiento: una época = recorrer train actualizando pesos + medir mAUC en validación.
    Guarda en `ruta` los pesos de la mejor época y se detiene si pasan `paciencia` épocas sin mejorar.
    Devuelve (mejor val mAUC, mejor época, historial [(loss_train, val_mAUC, lr)])."""
    opt = torch.optim.AdamW(_grupos(modelo, cfg['lr'], cfg['layer_decay']), weight_decay=cfg['wd'])
    plan = _Planificador(opt, cfg)
    perdida = nn.BCEWithLogitsLoss(pos_weight=None if pos_weight is None else pos_weight.to(_DISP))
    scaler = torch.amp.GradScaler(_DISP, enabled=_DISP == 'cuda')
    mejor, mejor_ep, sin_mejora, hist = 0., -1, 0, []
    for ep in range(cfg['epocas']):
        plan.inicio_epoca(ep)
        _modo_train(modelo, cfg)
        total = 0.
        for it, (X, y) in enumerate(tr):
            plan.iteracion(ep, it, len(tr))
            X = X.to(_DISP, non_blocking=True).contiguous(memory_format=torch.channels_last)
            with _amp():
                loss = perdida(modelo(X), y.to(_DISP))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        P, Y = _predecir(modelo, va)
        val = _mauc(_auc_por_clase(P, Y))
        hist.append((total / len(tr), val, plan.lr()))
        if verbose:
            print(f'ep {ep:2d} | loss train {total / len(tr):.4f} | val mAUC {val:.4f} | lr {plan.lr():.1e}')
        plan.fin_epoca(val)
        if val > mejor + 1e-4:
            mejor, mejor_ep, sin_mejora = val, ep, 0
            if ruta:
                torch.save(modelo.state_dict(), ruta)
        else:
            sin_mejora += 1
            if cfg['paciencia'] and sin_mejora >= cfg['paciencia']:
                if verbose:
                    print(f"parada temprana: {cfg['paciencia']} épocas sin mejorar")
                break
    return mejor, mejor_ep, hist


def _ajustar(datos, cfg, idx_train, idx_val, ruta=None, semilla=None, verbose=True):
    """Construye el modelo y los loaders y entrena. Es la ÚNICA función que une todas las piezas:
    la usan `entrenar` (train completo) y la búsqueda de hiperparámetros (subconjuntos)."""
    _semilla(cfg['semilla'] if semilla is None else semilla)
    tf_train, tf_eval = _transformaciones(cfg['aug']), _transformaciones('none')
    modelo = _crear_modelo(cfg)
    tr = _loader(datos, idx_train, cfg, tf_train, barajar=True)
    va = _loader(datos, idx_val, cfg, tf_eval, barajar=False)
    mejor, mejor_ep, hist = _bucle(modelo, tr, va, cfg, _pos_weight(datos, idx_train, cfg), ruta, verbose)
    if ruta:
        modelo.load_state_dict(torch.load(ruta, map_location=_DISP, weights_only=True))
    return modelo, hist, mejor, mejor_ep


@dataclass
class ModeloEntrenado:
    """Resultado de `entrenar` / `cargar_modelo`: el modelo (con los pesos de su mejor época) y todo lo que lo describe."""
    nombre: str
    cfg: dict
    modelo: nn.Module
    hist: list           # [(loss_train, val_mAUC, lr), ...] por época
    val_mAUC: float
    mejor_epoca: int
    test: dict = None    # lo rellena `resultados_test`
    probs: dict = field(default_factory=dict)   # caché de `probabilidades` por partición


def entrenar(datos, cfg, nombre=None, forzar=False, verbose=True):
    """Entrenamiento serio: todo el train, validación en cada época, guarda los pesos de la mejor época en
    `modelos/{nombre}.pt` y la descripción en `modelos/{nombre}_meta.pt`. Si el modelo ya está entrenado
    (existe su `_meta.pt`) lo carga en vez de reentrenar; usa forzar=True para repetir.
    Devuelve un `ModeloEntrenado`."""
    cfg = construir_config(cfg)
    nombre = nombre or _nombre_auto(cfg)
    if os.path.exists(_ruta(nombre, '_meta.pt')) and not forzar:
        print(f'{nombre}: ya entrenado, se carga de disco (forzar=True para repetir)')
        entrenado = cargar_modelo(nombre)
        if entrenado.cfg != cfg:
            print('  aviso: la configuración guardada difiere de la pedida:',
                  {k: (entrenado.cfg[k], cfg[k]) for k in cfg if entrenado.cfg.get(k) != cfg[k]})
        return entrenado
    modelo, hist, mejor, mejor_ep = _ajustar(datos, cfg, datos.train, datos.val, ruta=_ruta(nombre), verbose=verbose)
    entrenado = ModeloEntrenado(nombre, cfg, modelo, hist, mejor, mejor_ep)
    _guardar_meta(entrenado)
    print(f'{nombre}: {len(hist)} épocas, mejor la {mejor_ep + 1}, val mAUC {mejor:.4f}')
    return entrenado


# =====================================================================================================
# Guardar y cargar
# =====================================================================================================
def _ruta(nombre, extension='.pt'):
    """Ruta de un fichero de modelo dentro de la carpeta de modelos (se crea si no existe)."""
    os.makedirs(_DIR_MODELOS, exist_ok=True)
    return os.path.join(_DIR_MODELOS, nombre + extension)


def _guardar_meta(e):
    torch.save({'nombre': e.nombre, 'cfg': e.cfg, 'hist': e.hist, 'val_mAUC': e.val_mAUC,
                'mejor_epoca': e.mejor_epoca, 'test': e.test}, _ruta(e.nombre, '_meta.pt'))


def _leer_meta(nombre):
    ruta = _ruta(nombre, '_meta.pt')
    return torch.load(ruta, map_location='cpu', weights_only=False) if os.path.exists(ruta) else None


def cargar_modelo(nombre, cfg=None):
    """Carga de disco un modelo ya entrenado (`modelos/{nombre}.pt` + `modelos/{nombre}_meta.pt`), sin reentrenar.
    Si no hay `_meta.pt`, hay que pasar la configuración con `cfg`."""
    meta = _leer_meta(nombre)
    if cfg is None:
        if meta is None:
            raise FileNotFoundError(f'No existe {_ruta(nombre, "_meta.pt")}: pasa cfg=construir_config(...) con lo usado al entrenar')
        cfg = meta['cfg']
    cfg = construir_config(cfg)
    modelo = _crear_modelo({**cfg, 'init': None}, preentrenado=False)
    modelo.load_state_dict(torch.load(_ruta(nombre), map_location=_DISP, weights_only=True))
    modelo.eval()
    return ModeloEntrenado(nombre, cfg, modelo, meta['hist'] if meta else [], meta['val_mAUC'] if meta else float('nan'),
                           meta['mejor_epoca'] if meta else None, meta['test'] if meta else None)


# =====================================================================================================
# Resultados en test
# =====================================================================================================
def _evaluar_test(e, datos):
    P, Y = _predecir(e.modelo, _loader(datos, datos.test, e.cfg, _transformaciones('none'), barajar=False))
    aucs = _auc_por_clase(P, Y)
    nf = None
    if e.cfg['no_finding'] == 'derivado':     # AUROC de «No Finding» deducido de las 14 patologías
        sin_hallazgos = (Y.sum(1) == 0).float().numpy()
        nf = {k: float(roc_auc_score(sin_hallazgos, s.numpy())) for k, s in _puntuaciones_nf(P).items()}
    return {'test_mAUC': _mauc(aucs), 'aucs': aucs, 'nf_auc': nf, 'P_test': P[:, -_N_PATOLOGIAS:].clone()}


def _imprimir_test(e):
    t = PrettyTable(['Clase', 'AUROC test'])
    t.set_style(TableStyle.SINGLE_BORDER)
    aucs = e.test['aucs']
    for nombre, a in zip(ETIQUETAS[-len(aucs):], aucs.tolist()):
        t.add_row([nombre, f'{a:.4f}'])
    if e.test.get('nf_auc'):
        t.add_row(['No Finding (derivado, prod)', f"{e.test['nf_auc']['prod']:.4f}"])
        t.add_row(['No Finding (derivado, 1-max)', f"{e.test['nf_auc']['max']:.4f}"])
    t.add_divider()
    t.add_row(['mAUC (14 patologías)', f"{e.test['test_mAUC']:.4f}"])
    print(f'{e.nombre}: val mAUC {e.val_mAUC:.4f} | test mAUC {e.test["test_mAUC"]:.4f}')
    print(t)


def graficar_curvas(modelos, titulo=''):
    """Curvas de entrenamiento de uno o varios modelos (`ModeloEntrenado` o {nombre: ModeloEntrenado}): loss de
    train, mAUC de validación (con un punto en la mejor época) y learning rate. Sirve para ver el sobreajuste:
    la loss de train baja siempre, mientras la validación sube y después se estanca o cae."""
    if isinstance(modelos, ModeloEntrenado):
        modelos = {modelos.nombre: modelos}
    fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
    for nombre, m in modelos.items():
        ep = list(range(1, len(m.hist) + 1))
        ax[0].plot(ep, [h[0] for h in m.hist], label=nombre)
        linea, = ax[1].plot(ep, [h[1] for h in m.hist], label=nombre)
        if m.mejor_epoca is not None and m.mejor_epoca >= 0:
            ax[1].plot(m.mejor_epoca + 1, m.hist[m.mejor_epoca][1], 'o', color=linea.get_color())
        ax[2].plot(ep, [h[2] for h in m.hist], label=nombre)
    ax[0].set_title('Loss de train'); ax[1].set_title('mAUC de validación'); ax[2].set_title('Learning rate')
    ax[2].set_yscale('log')
    for a in ax:
        a.set_xlabel('época'); a.grid(alpha=.3)
    if len(modelos) > 1:
        ax[1].legend(fontsize=8)
    fig.suptitle(titulo or ', '.join(modelos))
    plt.tight_layout(); plt.show()


def _asegurar_test(e, datos, recalcular=False):
    """Evalúa en test si aún no se ha hecho (y guarda el resultado junto al modelo)."""
    if e.test is None or recalcular:
        e.test = _evaluar_test(e, datos)
        _guardar_meta(e)
    return e.test


def resultados_test(entrenado, datos, graficar=True, recalcular=False):
    """Evalúa en TEST (una sola vez: el resultado se guarda en `modelos/{nombre}_meta.pt` y las siguientes llamadas
    lo reutilizan, para no «mirar» el test repetidamente), imprime el AUROC por clase y dibuja las curvas de
    entrenamiento. Devuelve {'test_mAUC', 'aucs', 'nf_auc', 'P_test' (probabilidades de las 14 patologías)}."""
    _asegurar_test(entrenado, datos, recalcular)
    _imprimir_test(entrenado)
    if graficar and entrenado.hist:
        graficar_curvas(entrenado)
    return entrenado.test


def probabilidades(entrenado, datos, particion='test'):
    """Probabilidades (n, 14) de las patologías para 'train', 'val' o 'test'. Se calculan una vez y se guardan
    en el propio `ModeloEntrenado`. Sirve para promediar modelos, comparar, etc."""
    if particion not in entrenado.probs:
        if particion == 'test' and entrenado.test is not None and entrenado.test.get('P_test') is not None:
            entrenado.probs['test'] = entrenado.test['P_test']
        else:
            loader = _loader(datos, _indices(datos, particion), entrenado.cfg, _transformaciones('none'), barajar=False)
            P, _ = _predecir(entrenado.modelo, loader)
            entrenado.probs[particion] = P[:, -_N_PATOLOGIAS:]
    return entrenado.probs[particion]


# =====================================================================================================
# Búsqueda de hiperparámetros
# =====================================================================================================
def _subconjuntos(datos, n_train, n_val, semilla=42):
    """Mismos subconjuntos de train y validación en todas las pruebas, para que solo cambie el hiperparámetro."""
    rng = np.random.default_rng(semilla)
    return (rng.choice(datos.train, min(n_train, len(datos.train)), replace=False),
            rng.choice(datos.val, min(n_val, len(datos.val)), replace=False))


def _probar(datos, cfg, sub_train, sub_val, semilla=None):
    """Una prueba de la búsqueda: entrena con `cfg` y devuelve (mejor val mAUC, mejor época)."""
    modelo, _, mejor, mejor_ep = _ajustar(datos, cfg, sub_train, sub_val, semilla=semilla, verbose=False)
    del modelo
    if _DISP == 'cuda':
        torch.cuda.empty_cache()
    return mejor, mejor_ep


def _busqueda_voraz(probar, espacio):
    """Un parámetro cada vez, en el orden del diccionario: parte de la primera opción de cada lista, prueba todos
    los valores del parámetro actual con los demás fijos en su mejor valor hasta ahora, se queda con el mejor
    y pasa al siguiente. Barata (suma de opciones, no producto), pero NO explora combinaciones."""
    actual = {k: v[0] for k, v in espacio.items()}
    hechas, resultados = {}, []

    def valor(cambios):
        firma = repr(tuple(cambios[k] for k in espacio))
        if firma not in hechas:
            val, ep = probar(cambios)
            hechas[firma] = val
            resultados.append({**cambios, 'val_mAUC': val, 'mejor_epoca': ep})
            print(f'[{len(hechas)}] {cambios} -> val mAUC {val:.4f}')
        return hechas[firma]

    for clave, valores in espacio.items():
        actual[clave] = max(valores, key=lambda v: valor({**actual, clave: v}))
        print(f'>> mejor {clave}: {actual[clave]}')
    return resultados


def _busqueda_rejilla(probar, espacio):
    """Todas las combinaciones. Exhaustiva pero cara: nº de pruebas = producto del nº de opciones."""
    import itertools
    resultados = []
    combos = list(itertools.product(*espacio.values()))
    for i, valores in enumerate(combos, 1):
        cambios = dict(zip(espacio, valores))
        val, ep = probar(cambios)
        resultados.append({**cambios, 'val_mAUC': val, 'mejor_epoca': ep})
        print(f'[{i}/{len(combos)}] {cambios} -> val mAUC {val:.4f}')
    return resultados


def _tabla(filas, columnas, formato=None):
    t = PrettyTable(['#'] + columnas)
    t.set_style(TableStyle.SINGLE_BORDER)
    formato = formato or {}
    for i, f in enumerate(filas, 1):
        t.add_row([i] + [format(f[c], formato[c]) if c in formato else f[c] for c in columnas])
    print(t)


def buscar_hiperparametros(datos, base, espacio, metodo='voraz', n_train=25000, n_val=8000, epocas=4,
                           scheduler='coseno', confirmar=2, semillas=(1, 2, 3), nombre=None, forzar=False,
                           con_resultados=False):
    """Busca los mejores valores de `espacio` ({parámetro: [valores]}) partiendo de la configuración `base`.
    Cualquier parámetro de CONFIG_DEFECTO puede buscarse, incluido `capas`.

    Presupuesto de la búsqueda (para que sea rápida): subconjuntos de `n_train`/`n_val` imágenes y `epocas` épocas
    con el planificador `scheduler` ('coseno' converge en pocas épocas; 'plateau' necesita más). Todo lo demás
    sale de `base`, que además es lo que se devuelve con los parámetros elegidos, listo para `entrenar`.

    metodo      'voraz' (un parámetro cada vez, en el orden de `espacio`) o 'rejilla' (todas las combinaciones).
    confirmar   las N mejores se repiten con varias `semillas`; gana la de mayor media. Con 0 gana la mejor
                prueba directa, aunque una diferencia pequeña puede ser ruido (≈0,003 de mAUC entre semillas).
    nombre      si se da, guarda el resultado en `modelos/busqueda_{nombre}.pt`; si ya existe una búsqueda guardada
                con exactamente los mismos argumentos, la reutiliza en vez de repetirla (forzar=True para repetir).
    Devuelve la configuración completa elegida (y, con con_resultados=True, también un dict con las tablas)."""
    final = construir_config(base)
    busqueda = construir_config(final, epocas=epocas, scheduler=scheduler)
    for clave, valores in espacio.items():
        for v in valores:
            construir_config(busqueda, **{clave: v})              # valida claves y valores antes de empezar
    if metodo not in ('voraz', 'rejilla'):
        raise ValueError("metodo debe ser 'voraz' o 'rejilla'")
    firma = {'base': final, 'espacio': espacio, 'metodo': metodo, 'n_train': n_train, 'n_val': n_val,
             'epocas': epocas, 'scheduler': scheduler, 'confirmar': confirmar, 'semillas': tuple(semillas)}
    if nombre and os.path.exists(_ruta('busqueda_' + nombre)) and not forzar:
        guardada = torch.load(_ruta('busqueda_' + nombre), map_location='cpu', weights_only=False)
        if guardada.get('firma') == firma:
            print(f"búsqueda '{nombre}' ya hecha, se carga de disco (forzar=True para repetir)")
            print('Mejores parámetros:', {k: guardada['mejores'][k] for k in espacio})
            return (guardada['mejores'], guardada) if con_resultados else guardada['mejores']
        print(f"búsqueda '{nombre}' guardada con otros argumentos: se repite")

    sub_train, sub_val = _subconjuntos(datos, n_train, n_val)
    probar = lambda cambios, semilla=None: _probar(datos, {**busqueda, **cambios}, sub_train, sub_val, semilla)
    resultados = (_busqueda_voraz if metodo == 'voraz' else _busqueda_rejilla)(probar, espacio)
    resultados.sort(key=lambda r: r['val_mAUC'], reverse=True)
    print('\nResultados de la búsqueda (validación):')
    _tabla(resultados, list(espacio) + ['val_mAUC', 'mejor_epoca'], {'val_mAUC': '.4f'})
    ganador, confirmacion = resultados[0], []

    if confirmar:
        for r in resultados[:confirmar]:
            cambios = {k: r[k] for k in espacio}
            print(f'confirmando {cambios} con semillas {semillas}')
            vals = [probar(cambios, s)[0] for s in semillas]
            confirmacion.append({**cambios, 'media': float(np.mean(vals)), 'std': float(np.std(vals)), 'valores': vals})
        confirmacion.sort(key=lambda r: r['media'], reverse=True)
        print('\nConfirmación con varias semillas:')
        _tabla([{**c, 'valores': ', '.join(f'{v:.4f}' for v in c['valores'])} for c in confirmacion],
               list(espacio) + ['media', 'std', 'valores'], {'media': '.4f', 'std': '.4f'})
        ganador = confirmacion[0]

    mejores = {**final, **{k: ganador[k] for k in espacio}}
    print('\nMejores parámetros:', {k: mejores[k] for k in espacio})
    info = {'firma': firma, 'espacio': espacio, 'resultados': resultados, 'confirmacion': confirmacion, 'mejores': mejores}
    if nombre:
        torch.save(info, _ruta('busqueda_' + nombre))
    return (mejores, info) if con_resultados else mejores


# =====================================================================================================
# Comparar modelos
# =====================================================================================================
def _indices(datos, particion):
    return {'train': datos.train, 'val': datos.val, 'test': datos.test}[particion]


def etiquetas(datos, particion='test'):
    """Etiquetas (n, 14) de las patologías de 'train', 'val' o 'test', en el mismo orden que `probabilidades`."""
    return datos.Y[torch.as_tensor(_indices(datos, particion))][:, 1:]


def auc_por_clase(P, Y):
    """AUROC de cada patología. P e Y: tensores (n, 14) de probabilidades y etiquetas."""
    return _auc_por_clase(P, Y)


def mauc(P, Y):
    """mAUC: media del AUROC de las 14 patologías (la métrica de todo el trabajo)."""
    return _mauc(_auc_por_clase(P, Y))


def promediar(modelos, datos, particion='test'):
    """Promedio simple de las probabilidades (n, 14) de varios `ModeloEntrenado`. Cada modelo se equivoca de forma
    distinta; al promediar, los errores que no comparten se compensan. Sin pesos: nada se ajusta con validación."""
    return torch.stack([probabilidades(m, datos, particion) for m in modelos]).mean(0)


def _bootstrap_pacientes(datos, predicciones, Y, n_boot, semilla=0):
    """mAUC de cada modelo en `n_boot` remuestreos de PACIENTES de test (con reposición): las radiografías de un
    mismo paciente no son independientes, así que se remuestrean pacientes y no imágenes. Las mismas réplicas se
    usan para todos los modelos, lo que permite comparar diferencias en pareado."""
    pacientes = datos.paciente[datos.test]
    unicos, inverso = np.unique(pacientes, return_inverse=True)
    miembros = np.split(np.argsort(inverso, kind='stable'), np.cumsum(np.bincount(inverso))[:-1])
    rng = np.random.default_rng(semilla)
    salida = {k: np.empty(n_boot) for k in predicciones}
    for b in range(n_boot):
        sel = torch.from_numpy(np.concatenate([miembros[u] for u in rng.integers(0, len(unicos), len(unicos))]))
        for k, P in predicciones.items():
            salida[k][b] = _mauc(_auc_por_clase(P[sel], Y[sel]))
    return salida


def _ic95(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def _txt_ic(ic):
    return f'[{ic[0]:.4f}, {ic[1]:.4f}]'


def _resumen_modelo(lista):
    """(parámetros entrenables, 'épocas (mejor)') de un modelo suelto; '-' si es un promedio de varios."""
    if len(lista) != 1:
        return '-', '-'
    m = lista[0]
    params = sum(p.numel() for p in m.modelo.parameters() if p.requires_grad)
    epocas = f'{len(m.hist)} ({m.mejor_epoca + 1})' if m.hist and m.mejor_epoca is not None else '-'
    return f'{params:,}', epocas


def comparar(datos, candidatos, referencia=None, n_boot=200, por_clase=False, graficar=True):
    """Compara modelos (o promedios de modelos) con validación, test e intervalos de confianza.

    candidatos  {nombre: ModeloEntrenado} o {nombre: [ModeloEntrenado, ...]}. Con una lista se promedian las
                probabilidades de sus modelos (promedio de modelos).
    referencia  nombre contra el que se calcula la diferencia (Δ); por defecto, el primero.
    n_boot      réplicas del bootstrap por pacientes. IC 95 % = rango central del 95 % de esas réplicas. Si el
                intervalo de Δ incluye el 0, la diferencia con la referencia no se puede distinguir del ruido.
    por_clase   además, tabla del AUROC de cada patología.
    Devuelve {nombre: {'val_mAUC', 'test_mAUC', 'ic95', 'delta', 'ic95_delta', 'aucs'}}."""
    cand = {k: list(v) if isinstance(v, (list, tuple)) else [v] for k, v in candidatos.items()}
    referencia = referencia or next(iter(cand))
    Yv, Yt = etiquetas(datos, 'val'), etiquetas(datos, 'test')
    Pv = {k: promediar(v, datos, 'val') for k, v in cand.items()}
    Pt = {k: promediar(v, datos, 'test') for k, v in cand.items()}
    aucs = {k: _auc_por_clase(P, Yt) for k, P in Pt.items()}
    boot = _bootstrap_pacientes(datos, Pt, Yt, n_boot)

    res = {k: {'val_mAUC': _mauc(_auc_por_clase(Pv[k], Yv)), 'test_mAUC': _mauc(aucs[k]), 'ic95': _ic95(boot[k]),
               'delta': _mauc(aucs[k]) - _mauc(aucs[referencia]), 'ic95_delta': _ic95(boot[k] - boot[referencia]),
               'aucs': aucs[k]} for k in cand}

    t = PrettyTable(['Modelo', 'Nº modelos', 'Parámetros entrenables', 'Épocas (mejor)', 'val mAUC', 'test mAUC',
                     'IC 95 % test', f'Δ test vs {referencia}', 'IC 95 % Δ'])
    t.set_style(TableStyle.SINGLE_BORDER)
    for k, v in cand.items():
        params, epocas = _resumen_modelo(v)
        r, es_ref = res[k], k == referencia
        t.add_row([k, len(v), params, epocas, f"{r['val_mAUC']:.4f}", f"{r['test_mAUC']:.4f}", _txt_ic(r['ic95']),
                   '-' if es_ref else f"{r['delta']:+.4f}", '-' if es_ref else _txt_ic(r['ic95_delta'])])
    print(t)

    if por_clase:
        tc = PrettyTable(['Patología'] + list(cand))
        tc.set_style(TableStyle.SINGLE_BORDER)
        for i, nombre in enumerate(ETIQUETAS[1:]):
            tc.add_row([nombre] + [f'{aucs[k][i]:.4f}' for k in cand])
        tc.add_divider()
        tc.add_row(['mAUC'] + [f"{res[k]['test_mAUC']:.4f}" for k in cand])
        print(tc)

    if graficar:
        graficar_resultados({'test mAUC': {k: res[k]['test_mAUC'] for k in cand}},
                            titulo='mAUC en test (IC 95 % por pacientes)', ylabel='test mAUC', horizontal=True,
                            referencia={'azar': 0.5}, errores={'test mAUC': {k: res[k]['ic95'] for k in cand}},
                            ylim=(0.5, min(1.0, max(r['ic95'][1] for r in res.values()) + 0.04)))
    return res


def comparar_no_finding(datos, pares, graficar=True):
    """«No Finding» como clase (15 salidas) frente a «No Finding» derivado (14 salidas: se deduce de que no haya
    ninguna patología). pares: {nombre: (modelo_con_clase, modelo_derivado)}. El mAUC de las 14 patologías es
    comparable entre ambos; el AUROC de «No Finding» sale de su propia salida (clase) o de las puntuaciones
    'prod' = prod(1 - p) y '1-max' = 1 - max(p) (derivado). Evalúa en test los modelos que aún no lo estén."""
    t = PrettyTable(['Estrategia', 'val mAUC (con NF)', 'val mAUC (sin NF)', 'test mAUC (con NF)', 'test mAUC (sin NF)',
                     'AUROC NF (clase)', 'AUROC NF (prod)', 'AUROC NF (1-max)'])
    t.set_style(TableStyle.SINGLE_BORDER)
    mauc_ = {'val, con NF': {}, 'val, sin NF': {}, 'test, con NF': {}, 'test, sin NF': {}}
    nf = {'clase (15 salidas)': {}, 'derivado: prod(1-p)': {}, 'derivado: 1-max(p)': {}}
    for nombre, (clase, derivado) in pares.items():
        tc, td = _asegurar_test(clase, datos), _asegurar_test(derivado, datos)
        auc_clase = tc['aucs'][0].item()
        t.add_row([nombre, f'{clase.val_mAUC:.4f}', f'{derivado.val_mAUC:.4f}', f"{tc['test_mAUC']:.4f}",
                   f"{td['test_mAUC']:.4f}", f'{auc_clase:.4f}', f"{td['nf_auc']['prod']:.4f}", f"{td['nf_auc']['max']:.4f}"])
        for clave, valor in zip(mauc_, (clase.val_mAUC, derivado.val_mAUC, tc['test_mAUC'], td['test_mAUC'])):
            mauc_[clave][nombre] = valor
        for clave, valor in zip(nf, (auc_clase, td['nf_auc']['prod'], td['nf_auc']['max'])):
            nf[clave][nombre] = valor
    print(t)
    if graficar:
        graficar_resultados(mauc_, titulo='mAUC de las 14 patologías, con y sin «No Finding» como clase', ylim=(0.5, 0.95))
        graficar_resultados(nf, titulo='AUROC de «No Finding» (test)', ylabel='AUROC', ylim=(0.5, 0.95))


# =====================================================================================================
# Gráficas
# =====================================================================================================
_PALETA = ['lightskyblue', 'steelblue', 'moccasin', 'darkorange', 'mediumseagreen', 'seagreen', 'lightgray', 'salmon']


def graficar_resultados(datos, titulo='', ylabel='mAUC', ylim=None, colores=None, errores=None, referencia=None,
                        horizontal=False, rotar=0, valores=None, leyenda=True, ax=None):
    """Barras agrupadas: la gráfica de resultados de uso general.
      datos       {serie: {categoría: valor}}. Cada serie es un color; las categorías (modelos, grupos, clases...)
                  salen del primer dict. Con una sola serie son barras simples.
      errores     {serie: {categoría: (bajo, alto)}} con los extremos ABSOLUTOS de un intervalo (p. ej. IC 95 %).
      referencia  número o {etiqueta: valor}: línea discontinua (azar, media global...).
      horizontal  barras horizontales (útil con nombres de categoría largos).
      valores     escribir el valor sobre cada barra. Por defecto solo si hay 16 barras o menos (si no, se pisan).
      ax          si se pasa, dibuja en ese eje para componer varias gráficas en una figura; entonces la figura
                  la muestras tú (plt.tight_layout(); plt.show())."""
    creado = ax is None
    if creado:
        _, ax = plt.subplots(figsize=(9, 4.2))
    series, cats = list(datos), list(next(iter(datos.values())))
    colores = colores or _PALETA
    n, w, pos = len(series), 0.8 / len(series), np.arange(len(cats))
    if valores is None:
        valores = n * len(cats) <= 16
    for i, s in enumerate(series):
        v = np.array([datos[s][c] for c in cats], float)
        off = (i - (n - 1) / 2) * w
        lo = hi = np.zeros(len(cats))
        err = None
        if errores and s in errores:
            lo = np.array([max(v[j] - errores[s][c][0], 0) for j, c in enumerate(cats)])
            hi = np.array([max(errores[s][c][1] - v[j], 0) for j, c in enumerate(cats)])
            err = [lo, hi]
        kw = dict(color=colores[i % len(colores)], edgecolor='black', label=s if n > 1 else None)
        if horizontal:
            ax.barh(pos + off, v, w, xerr=err, capsize=3, **kw)
        else:
            ax.bar(pos + off, v, w, yerr=err, capsize=3, **kw)
        if valores:
            for p, val, h in zip(pos + off, v, hi):
                if horizontal:
                    ax.annotate(f'{val:.3f}', (val + h, p), xytext=(3, 0), textcoords='offset points', va='center', fontsize=8)
                else:
                    ax.annotate(f'{val:.3f}', (p, val + h), xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)
    if horizontal:
        ax.set_yticks(pos); ax.set_yticklabels(cats); ax.invert_yaxis(); ax.set_xlabel(ylabel)
        if ylim:
            ax.set_xlim(*ylim)
        ax.grid(axis='x', alpha=.3)
    else:
        ax.set_xticks(pos); ax.set_xticklabels(cats, rotation=rotar, ha='right' if rotar else 'center'); ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis='y', alpha=.3)
    if referencia is not None:
        for etiqueta, valor in (referencia if isinstance(referencia, dict) else {'referencia': referencia}).items():
            (ax.axvline if horizontal else ax.axhline)(valor, color='gray', ls='--', label=etiqueta)
    if leyenda and (n > 1 or referencia is not None):
        ax.legend(fontsize=8)
    ax.set_title(titulo)
    if creado:
        plt.tight_layout(); plt.show()
    return ax
