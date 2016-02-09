import os
import argparse
import csv
from predict import predict


def evaluate(input_csv, proto, model):

  os.rmdir("tmp")
  os.mkdir("tmp")

  correct_files = []
  incorrect_files = []

  reader = csv.reader(file(input_csv, 'rU'))
  for filename, label in reader:

    prediction = predict(filename, proto, model, "tmp")

    if prediction[0].argmax == label:
      correct_files.append(filename)
    else:
      incorrect_files.append(filename)

    print 'predicted class:', prediction[0].argmax()

  os.rmdir("tmp")

  num_correct = len(correct_files)
  num_incorrect = len(incorrect_files)
  print "Correctly Classified: {0} ({ratio}%)".format(num_correct, num_correct / (num_correct + num_incorrect))
  print "Incorrectly Classified: {0} ({ratio}%)".format(num_correct, num_incorrect / (num_correct + num_incorrect))

  # Save correct / incorrect filenames in txt file

  correct_out = open("correct_files.txt", "wb")
  incorrect_out = open("correct_files.txt", "wb")

  correct_out.write("\n".join(itemlist))
  incorrect_out.write("\n".join(itemlist))

  correct_out.close()
  incorrect_out.close()


if __name__ == '__main__':

  argparser = argparse.ArgumentParser()
  argparser.add_argument("--csv", required=True)
  argparser.add_argument("--proto", required=True)
  argparser.add_argument("--model", required=True)

  args = argparser.parse_args()
  evaluate(args.csv, args.proto, args.model)