from dataclasses import dataclass, field
import logging
from os.path import join, isabs, exists

from util import (
    get_current_ts,
    create_dir_if_not_exists,
    get_root_path,
    string_to_underscore
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentArguments:
    """
    Arguments related to the dataset
    """
    dataset_name: str = field(metadata={
        'help': 'The name of the dataset'
    })

    market: str = field(
        metadata={'help': 'The market if classification'}
    )

    metapaths: list = field(
        default=None,
        metadata={'help': 'The metapaths to use'}
    )

    market_source: str = field(
        default='diffbot',
        metadata={'help': 'The source of the market labels',
                  'choices': ['diffbot']}
    )

    experiment_name: str = field(
        default='default',
        metadata={'help': 'The name of the experiment'}
    )

    debug: str = field(
        default=False,
        metadata={'help': 'Whether to use debug logging'}
    )

    node_embds_path: str = field(
        default=None,
        metadata={'help': 'The path to the text node embeddings if any. Can be an absolute path'
                          'or path starting from the root of project directory'}
    )

    natts_path: str = field(
        default=None,
        metadata={
        'help': 'The path to the node attribute specifications'
    })

    task: str = field(
        default='binary_classification',
        metadata={'help': 'The training and testing task'}
    )

    model: str = field(
        default='hrgcn',
        metadata={'help': 'The name of the model to use. Default is magnn',
                  'choices': ['han', 'hrgcn', 'mlp']}
    )

    def __post_init__(self):
        market_name = '_'.join(self.market.lower().replace('-', ' ').split())
        exp_str = string_to_underscore(self.experiment_name)
        save_path_name = f'{exp_str}_{self.dataset_name}_{market_name}_{self.model}'
        self.log_dir = join(get_root_path(), 'logs', save_path_name + f'_{get_current_ts()}')
        create_dir_if_not_exists(self.log_dir)
        if self.node_embds_path is not None:
            self.node_embds_path = join(get_root_path(), self.node_embds_path)
        if self.natts_path is not None:
            if not isabs(self.natts_path):
                self.natts_path = self.natts_path.lstrip('.')
                self.natts_path = join(get_root_path(), self.natts_path)
            assert exists(self.natts_path), f"The natts path {self.natts_path} does not exist"
        if self.model == 'han':
            assert self.metapaths is not None, 'Need to specify metapaths for han model'

@dataclass
class TrainingArguments:
    num_layers: str = field(
        default=2,
        metadata={'help': 'Number of layers. Default is 2'})
    device: int = field(
        default=0,
        metadata={'help': 'The cuda device'}
    )
    hidden_dim: int = field(
        default=64,
        metadata={'help': 'Dimension of the node hidden state. Default is 64.'}
    )
    num_heads: int = field(
        default=8,
        metadata={'help': 'Number of the attention heads. Default is 8.'}
    )
    num_epochs: int = field(
        default=100,
        metadata={'help': 'Number of epochs. Default is 100'}
    )
    patience: int = field(
        default=10,
        metadata={'help': 'How long to wait after last time validation loss improved. Default is 10'}
    )
    repeat: int = field(
        default=1,
        metadata={'help': 'Repeat the training and testing for N times. Default is 1.'}
    )
    dropout_rate: float = field(
        default=0.5
    )
    val_metric: str = field(
        default='f1',
        metadata={'help': 'The validation metric in which early stopping is determined'}
    )

    lr: float = field(
        default=0.005
    )
    weight_decay: float = field(
        default=0.001
    )

    print_val_epochs: int = field(
        default=10,
        metadata={'help': 'Number of epochs betweeen printing validation results'}
    )