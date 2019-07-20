from utils.parse_args import parse_args
from models.pcn import *

# Getting the arguments
args = parse_args()
# Loading the model
model = STN3d(args)

model.eval()


