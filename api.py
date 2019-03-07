'''
NLU api (rasa and rnn)
Author: Heng-Da Xu <dadamrxx@gmail.com>
Modify: 2/23/2019
'''
import logging
import os
import typing

import easyto.intent_classifier.main
import rasa_nlu.model
import spacy
import thulac

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('[%(name)s] [%(asctime)s] - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class RasaNLU:

    def __init__(self, model_dir, chitchat_threshold=0.1):
        names = os.listdir(model_dir)
        self.interpreters = {}
        for index, name in enumerate(names):
            logger.info('RasaNLU - loading model (%d/%d): %s' % (index + 1, len(names), name))
            model_path = os.path.join(model_dir, name)
            self.interpreters[name] = rasa_nlu.model.Interpreter.load(model_path)
        self.chitchat_threshold = chitchat_threshold

    def parse(self, text):
        logger.info('RasaNLU - text: %s' % text)

        # Make sure 'scene' model is loaded
        if 'scene' not in self.interpreters.keys():
            raise KeyError('"scene" model is not found. The existing models:',
                            list(self.interpreters.keys()))
        scene_result = self.interpreters['scene'].parse(text)
        intent = scene_result['intent']
        scene, confidence = intent['name'], intent['confidence']
        logger.info('RasaNLU - got scene: %s (confidence: %.2f)' % (scene, confidence))

        # Make sure the scene is in model list
        if scene not in self.interpreters.keys():
            raise KeyError('scene "%s" is not in loaded models. The existing models:' % scene,
                            list(self.interpreters.keys()))
        result = self.interpreters[scene].parse(text)
        logger.info('RasaNLU - got intent: %s (confidence: %.2f)' %
                      (result['intent']['name'], result['intent']['confidence']))

        # If intent confidence is lower than threshold, the chit_chat will be returned
        if result['intent']['confidence'] < self.chitchat_threshold:
            intent_ranking = result['intent_ranking']
            result = {
                'entities': [],
                'intent': {
                    'name': 'chit_chat',
                    'confidence': 1.0,
                },
                'intent_ranking': intent_ranking,
                'text': text,
            }
        respose = {
            'model': 'rasa',
            'scene': scene,
            'scene_result': scene_result,
            'nlu_result': result,
        }
        return respose


class RnnNLU:

    def __init__(self, model_dir, chitchat_threshold=0.1):
        names = {
            'scene': 'global_rnn_pos_target.h5',
            'ac': 'ac_rnn_pos_target.h5',
            'navi': 'navi_rnn_pos_target.h5',
            'music': 'music_rnn_pos_target.h5',
            'radio': 'radio_rnn_pos_target.h5',
            'wiper': 'wiper_rnn_pos_target.h5',
            'mail': 'mail_rnn_pos_target.h5',
            'wechat': 'wechat_rnn_pos_target.h5',
            'driven': 'driven_rnn_pos_target.h5',
            'vehicle': 'vehicle_rnn_pos_target.h5',
            'weather': 'weather_rnn_pos_target.h5',
            'light': 'light_rnn_pos_target.h5',
        }
        self.interpreters = {}
        for index, (key, name) in enumerate(names.items()):
            model_path = os.path.join(model_dir, name)
            logger.info('RnnNLU - loading model (%d/%d): %s' % (index + 1, len(names), key))
            self.interpreters[key] = easyto.intent_classifier.main.Model(model_path, mode=key)
        self.cuter = thulac.thulac()
        self.chitchat_threshold = chitchat_threshold

    def parse(self, text):
        logger.info('RnnNLU - text: %s' % text)

        # Make sure 'scene' model is loaded
        if 'scene' not in self.interpreters.keys():
            raise KeyError('"scene" model is not found. The existing models:',
                            list(self.interpreters.keys()))
        intent = self.interpreters['scene'].predict(text, self.cuter)
        scene_result = {
            'name': intent['intent'],
			'confidence': float(intent['confidence']),
        }
        scene, confidence = intent['intent'], float(intent['confidence'])
        logger.info('RasaNLU - got scene: %s (confidence: %.2f)' % (scene, confidence))

        # Make sure the scene is in model list
        if scene not in self.interpreters.keys():
            raise KeyError('scene "%s" is not in loaded models. The existing models:' % scene,
                            list(self.interpreters.keys()))
        rnn_result = self.interpreters[scene].predict(text, self.cuter)
        result = {
            'entities': [],
            'intent': {
                'name': rnn_result['intent'],
                'confidence': float(rnn_result['confidence']),
            },
            'intent_ranking': [],
            'text': text,
        }
        logger.info('RnnNLU - got intent: %s (confidence: %.2f)' %
                      (result['intent']['name'], result['intent']['confidence']))

        if result['intent']['confidence'] < self.chitchat_threshold:
            result['intent']['name'] = 'chit_chat'
            result['intent']['confidence'] = 1.0
        respose = {
            'model': 'rnn',
            'scene': scene,
            'scene_result': scene_result,
            'nlu_result': result,
        }
        return respose


class Ner:

    def __init__(self, model_dir:str):
        names = os.listdir(model_dir)
        self.models = {}
        for index, name in enumerate(names):
            logger.info('Ner - loading model (%d/%d): %s' % (index + 1, len(names), name))
            model_path = os.path.join(model_dir, name)
            self.models[name] = spacy.load(model_path)

    def parse(self, model_name:str, text:str):
        logger.info('Ner - model name: %s' % model_name)
        if model_name not in self.models.keys():
            return []
        doc = self.models[model_name](text)
        entities = []
        for ent in doc.ents:
            logger.info('Ner - value: %s, entity: %s' % (ent.text, ent.label_))
            d = {
                'value': ent.text,
                'entity': ent.label_,
            }
            entities.append(d)
        return entities


class HybridNLU:

    def __init__(self, models=None):
        if isinstance(models, str):
            models = [models]
        assert isinstance(models, typing.Iterable)
        self.models = models

        if 'rasa' in models:
            self.rasa_nlu = RasaNLU('rasa_models')
        if 'rnn' in models:
            self.rnn_nlu = RnnNLU('easyto/intent_classifier/models')
        if 'ner' in models:
            self.ner = Ner('ner_models')

    def parse(self, data):
        assert data.get('model') is not None
        assert data['model'] in ['rasa', 'rnn']
        assert data.get('text') is not None
        assert isinstance(data['text'], str)

        if data['model'] == 'rasa':
            respose = self.rasa_nlu.parse(data['text'])
        elif data['model'] == 'rnn':
            respose = self.rnn_nlu.parse(data['text'])
        if 'ner' in self.models:
            scene = respose['scene']
            respose['entities']['nlu_result'] = self.ner.parse(scene, data['text'])
        return respose
