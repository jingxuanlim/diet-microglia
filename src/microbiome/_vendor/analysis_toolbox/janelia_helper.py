# Vendored from analysis_toolbox (janelia_helper.py) at commit 7f96772069025d136f802c364879703b086dec94 (plus uncommitted working-tree changes).
# Only definitions reached by this project are included; see _vendor/README.md.


nrs_data = "/nrs/ahrens/jing/"


dm11_data = "/groups/ahrens/ahrenslab/jing/"


dm11_home = "/groups/ahrens/home/limj2/"


def janelia_path(directory, server=''):

    import pathlib

    ## check if directory is windows or posix
    if directory[0] != '/':
        path = pathlib.PureWindowsPath(directory)

        if server == 'nrs':
            full_path = pathlib.Path(nrs_data + '/' + \
                                     path.relative_to(path.drive).as_posix())
        elif server == 'dm11 data':
            full_path = pathlib.Path(dm11_data + '/' + \
                                     path.relative_to(path.drive).as_posix())
        elif server == 'dm11 home':
            full_path = pathlib.Path(dm11_home + '/' + \
                                     path.relative_to(path.drive).as_posix())            

        else:
            full_path = path

    else: 
        path = pathlib.PosixPath(directory)
        full_path = path

    return full_path
