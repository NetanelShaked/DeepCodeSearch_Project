from CNN_model import CNN_Model
from IGTD import create_Pictures
from CodeBert import script2

# script2.run('./CodeBert/cosqa-train.json', './CodeBert/cosqa-train.csv')
# create_Pictures.run('./CodeBert/cosqa-train.csv', './IGTD/Data')
# CNN_Model.train('./IGTD/Data', './CodeBert/cosqa-train.csv', './CNN_model/model')

# script2.run('./CodeBert/cosqa-dev.json', './CodeBert/cosqa-test.csv')
create_Pictures.run('./CodeBert/cosqa-test.csv', './IGTD/Dev')

model_path = './CNN_model/model2'
model = CNN_Model.loadModel(model_path)

predict_res = CNN_Model.predict('./IGTD/Dev1', model)
print(predict_res)