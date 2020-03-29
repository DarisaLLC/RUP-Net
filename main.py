import sys
import argparse
from util import *
from RUP_Net import *


#pay attention: "_" in main
#have to put parser in main(_), not __main__
#otherwise, tensorflow will cause error "UnrecognizedFlagError: Unknown command line flag"
def main():
    parser = argparse.ArgumentParser(description="Residual U-Net + Pixel Net",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--debug', action="store_true", help="output debug info")
    parser.add_argument('-j', '--json_file', type=str, default='./json/train.json',
                        help='json file with network configurations')
    if sys.__stdin__.isatty():
        args = parser.parse_args()
    else:
        args = parser.parse_args([])

    print(args)

    hyperParams = util.HyperParams(args.json_file).dict
    net = RUP_Net(hyperParams=hyperParams)
    net.model.summary()
    getattr(net, hyperParams['MODE'])()


if __name__ == "__main__":
    #tf.app.run()
    main()
