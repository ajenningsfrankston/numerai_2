from numerapi import NumerAPI
import pickle
import os



id = "OML65REYFDPC5O7N22XCRP44BG2M74XH"
key = "YSTL455VERL7WZ4D7OQ6XEYEQN2MRCCICBMILNFP3DUZC4MSAS2WSH2MV7ED6WB3"

api = NumerAPI(public_id=id,secret_key=key)

base_path = "../../numerai_predictions/"

competitions = api.get_tournaments()
for comp in competitions:
    name = comp['name']
    path = base_path + name + ".csv"
    print('uploading ' + name)
    api.upload_predictions(path,comp['tournament'])

