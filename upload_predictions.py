from numerapi import NumerAPI
import pickle

id = "OML65REYFDPC5O7N22XCRP44BG2M74XH"
key = "YSTL455VERL7WZ4D7OQ6XEYEQN2MRCCICBMILNFP3DUZC4MSAS2WSH2MV7ED6WB3"

api = NumerAPI(public_id=id,secret_key=key)

plist_filename = "../../numerai_predictions/" + "prediction_list"

with open(plist_filename,'rb') as fp:
    path_list = pickle.load(fp)

i = 1
for path in path_list :
    api.upload_predictions(path,i)
    i = i+1
