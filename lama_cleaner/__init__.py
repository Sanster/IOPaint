from lama_cleaner.parse_args import parse_args
from lama_cleaner.server import main


def entry_point():
    args = parse_args()
    main(args)
