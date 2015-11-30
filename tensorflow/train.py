import yaml
import networkinput
from network.instances.berlinnet import berlin_net

config = yaml.load(file("config.yaml"))

training_set = networkinput.CSVInput(config['TRAINING_DATA'], config['INPUT_SHAPE'], config['OUTPUT_SHAPE'][0], mode="L")
test_set = networkinput.CSVInput(config['TEST_DATA'], config['INPUT_SHAPE'], config['OUTPUT_SHAPE'][0], mode="L")

berlin_net.add_training_input(training_set, test_set)
berlin_net.set_cost()
berlin_net.set_optimizer(config['LEARNING_RATE'])
berlin_net.set_accuracy()
berlin_net.set_log_path(config['LOG_PATH'])
berlin_net.set_snapshot_path(config['SNAPSHOT_PATH'])

berlin_net.run(config['BATCH_SIZE'], config['TRAINING_ITERS'], config['DISPLAY_STEP'])

