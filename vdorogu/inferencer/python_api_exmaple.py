from vdorogu.inferencer.inference import Inferencer

inf = Inferencer(model="inferencer/models/web/xlm_roberta_large_assessor", storage="./models", debug=True, gpus=4)

with open("input_file.txt") as fin:
    scores = inf.inference_text(fin)
    # or
    for res in inf.inference_text_realtime(fin):
        print(res)

# or
scores = inf.fields_inference([('abc', 'cde'), ('abc', 'efg')])
