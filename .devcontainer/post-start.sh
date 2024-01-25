# Setup lefthook
git init
lefthook install

# Disable Graphite pager
gt user pager --disable

# Install dependencies
pip install -e ".[dev, test]"

