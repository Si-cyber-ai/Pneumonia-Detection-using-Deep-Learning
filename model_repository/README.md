# Triton model repository — managed by scripts/convert_models.py
#
# Run:  python scripts/convert_models.py
#
# Generated structure after conversion:
#   model_repository/
#   ├── pneumo_densenet/
#   │   ├── config.pbtxt          ← auto-generated with correct tensor names
#   │   ├── tensor_names.json     ← metadata used by backend
#   │   └── 1/
#   │       ├── saved_model.pb
#   │       └── variables/
#   └── pneumo_resnet/
#       ├── config.pbtxt
#       ├── tensor_names.json
#       └── 1/
#           ├── saved_model.pb
#           └── variables/
