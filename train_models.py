import json
from pathlib import Path

from models import models
from util import get_dataset

epochs = 20

save_folder = Path('models_data')
train_cycles = 5
if __name__ == '__main__':
    train = get_dataset('data/train')
    validation = get_dataset('data/validate')

    act: str
    hls: int
    for act in models:
        for hls in models[act]:
            for i in range(train_cycles):
                model = models[act][hls]()
                history = model.fit(train, validation_data=validation, epochs=epochs, batch_size=32)

                save_path = Path(save_folder, act, str(hls), str(i + 1))
                save_path.mkdir(parents=True, exist_ok=True)

                model.save_weights(Path(save_path, 'model.weights.h5'))
                with open(Path(save_path, 'history.json'), 'w') as file:
                    json.dump(history.history, file)
