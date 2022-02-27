from CNN_model import CNN_Model
from IGTD import create_Pictures
from CodeBert import script2

# script2.run('./CodeBert/cosqa-train.json', './CodeBert/cosqa-train2.csv')
# create_Pictures.run('./CodeBert/cosqa-train2.csv', './IGTD/Data2')
# CNN_Model.train('./IGTD/Data2', './CodeBert/cosqa-train2.csv', './CNN_model/model2')

# script2.run('./CodeBert/cosqa-dev.json', './CodeBert/cosqa-test2.csv')
create_Pictures.run('./CodeBert/cosqa-test2.csv', './IGTD/Dev2')

model_path = './CNN_model/model2'
model = CNN_Model.loadModel(model_path)

predict_res = CNN_Model.predict('./IGTD/Dev2', model)
print(predict_res)