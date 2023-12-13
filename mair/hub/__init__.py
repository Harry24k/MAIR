import warnings

import torch

from .. import RobModel
from ..utils import load_model
from ._parse_notion import get_link_by_id


def load_pretrained(id, flag, save_dir="./"):
    assert flag in ['Init', 'Best', 'Last']
    url = get_link_by_id(id, flag)
    state_dict = torch.hub.load_state_dict_from_url(url,
                                                    progress=True, model_dir=save_dir,
                                                    file_name="%s_%s.pth"%(id, flag),
                                                    map_location='cpu')
    return _construct_rmodel_from_dict(id, state_dict)


def _construct_rmodel_from_dict(id, state_dict):
    model_name = id.split("_")[1]
    n_classes = state_dict['rmodel']['n_classes'].item()
    model = load_model(model_name=model_name, n_classes=n_classes)
    rmodel = RobModel(model, n_classes=n_classes)
    rmodel.load_state_dict_auto(state_dict['rmodel'])
    return rmodel
