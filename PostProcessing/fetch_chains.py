from unimpeded.database import DatabaseExplorer

# Initialise DatabaseExplorer
dbe = DatabaseExplorer()
# Get a list of currently available models and datasets
models_list = dbe.models
datasets_list = dbe.datasets
# # Choose model, dataset and sampling method
# method = 'ns' # 'ns' for nested sampling, 'mcmc' for MCMC
# model = "klcdm" # from models_list
# dataset = "des_y1.joint+planck_2018_CamSpec" # from datasets_list
# # Download samples chain
# samples = dbe.download_samples(method, model, dataset)
# # Download Cobaya and PolyChord run settings
# info = dbe.download_info(method, model, dataset)
print(models_list)
