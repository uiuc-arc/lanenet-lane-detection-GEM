from pathlib import Path

from lanenet.parse_config_utils import Config


THIS_DIR = Path(__file__).parent

config_path = THIS_DIR / 'conf_tusimple_lanenet.yaml'

lanenet_cfg = Config(config_path=config_path)
