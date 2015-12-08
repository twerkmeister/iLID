import yaml
import networkinput
from network.instances.berlinnet import net

config = yaml.load(file("config.yaml"))

training_set = networkinput.CSVInput(config['TRAINING_DATA'], config['INPUT_SHAPE'], config['OUTPUT_SHAPE'][0], mode="L")
test_set = networkinput.CSVInput(config['TEST_DATA'], config['INPUT_SHAPE'], config['OUTPUT_SHAPE'][0], mode="L")

net.set_training_input(training_set, test_set)
net.set_cost()
net.set_optimizer(config['LEARNING_RATE'], config['LEARNING_DECAY_STEP'])
net.set_accuracy()
net.set_log_path(config['LOG_PATH'])
net.set_snapshot_path(config['SNAPSHOT_PATH'])

net.train(config['BATCH_SIZE'], config['TRAINING_ITERS'], config['DISPLAY_STEP'])

