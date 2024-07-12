INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["What is quantum computing?"]
    },
    "temperature": {
        'datatype': 'FP32',
        'required': True,
        'shape': [1],
        'example': [0.7]
    },
    "top_p": {
        'datatype': 'FP32',
        'required': True,
        'shape': [1],
        'example': [0.1]
    },
    "repetition_penalty": {
        'datatype': 'FP32',
        'required': True,
        'shape': [1],
        'example': [1.18]
    },
    "max_tokens": {
        'datatype': 'INT8',
        'required': True,
        'shape': [1],
        'example': [512]
    }
}
