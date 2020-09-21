from tools.translate import build_model, translate, translate_beam_search, process_input, predict

import torch

class Predictor():
    def __init__(self, config):

        device = config['device']
        
        model, vocab = build_model(config)
        weights = 'weights/transformerocr.pth'

        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))

        self.config = config
        self.model = model
        self.vocab = vocab
        

    def predict(self, img):
        img = process_input(img, self.config['dataset']['image_height'], 
                self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])   

        img = img.to(self.config['device'])

        if self.config['predictor']['beamsearch']:
            sent = translate_beam_search(img, self.model)
            s = sent
        else:
            sents = translate(img, self.model)
            s = translate(img, self.model)[0].tolist()

        s = self.vocab.decode(s)

        return s

def get_text(recognizer,image_list):
    coord = [item[0] for item in image_list]
    img_list = [item[1] for item in image_list]
    result_str = []
    for img in img_list:
        result_str.append(recognizer.predict(img))
    result = []
    for box,rs in zip(coord,result_str):
        result.append((box,rs))
    return result