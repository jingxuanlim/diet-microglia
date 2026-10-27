# Vendored from analysis_toolbox (list.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
