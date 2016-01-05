import yaml
import argparse
import importlib
import networkinput
from util.email_notification import send_email_notification

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default="config.yaml", help="Path to config file")
args = parser.parse_args()

config = yaml.load(file(args.config))

network_module = "network.instances.{0}".format(config["NET"])
network_generator = getattr(importlib.import_module(network_module), "generate")

net = network_generator(config["INPUT_SHAPE"], config["OUTPUT_SHAPE"][0])

training_set = networkinput.CSVInput(config["TRAINING_DATA"], config["INPUT_SHAPE"], config["OUTPUT_SHAPE"][0], mode="L")
test_set = networkinput.CSVInput(config["TEST_DATA"], config["INPUT_SHAPE"], config["OUTPUT_SHAPE"][0], mode="L")

net.set_training_input(training_set, test_set)
net.set_cost()
net.set_optimizer(config["LEARNING_RATE"], config["LEARNING_DECAY_STEP"])
net.set_accuracy()
net.set_log_path(config["LOG_PATH"])
net.set_snapshot_path(config["SNAPSHOT_PATH"])

precision = net.train(config["BATCH_SIZE"], config["TRAINING_ITERS"], config["DISPLAY_STEP"])

if config["ENABLE_EMAIL_NOTIFICATION"]:
  send_email_notification(precision)

