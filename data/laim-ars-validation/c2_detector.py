from typing         import Literal
from dataclasses    import dataclass, field

from pathlib        import PurePath

from tui  import ColorSchemeDataScience

from meta       import S1Meta


_METRIC_T = Literal['accuracy', 'precision', 'recall', 'specificity', 'f1', 'youden']


@dataclass(frozen = True)
class S2Config:
    '''конфигурация этапа обучения детектора'''

    s1_meta                     : S1Meta
    output_dir                  : PurePath
    output_prefix               : str
    run_id                      : None | str

    threshold_metric            : _METRIC_T = 'youden'
    select_metric               : _METRIC_T = 'youden'

    n_thresholds                : int   = 5000
    inference_normal_count      : int   = 5000
    inference_anomalous_count   : int   = 5000

    seed                        : int   = 12345
    eps                         : float = 1e-8

    device                      : str   = 'cpu'

    output_color_scheme         : ColorSchemeDataScience = field(
        default_factory = ColorSchemeDataScience)

    def _prefix_for(self) -> str:
        return f'{self.output_prefix}_{self.run_id}' if self.run_id else self.output_prefix