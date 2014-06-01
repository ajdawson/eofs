"""Documentation generation script.

Copies the built html documentation into the gh-pages branch of the
appropriate repository and makes a commit.

The commit needs to be pushed manually after it has been verified.

"""
import os
import re
import sys
from os import chdir as cd
from os.path import join as pjoin

from subprocess import Popen, PIPE, CalledProcessError, check_call


# ----------------------------------------------------------------------------
# Globals
#
pages_dir = 'gh-pages'
html_dir = '_build/html'
pages_repo = 'git@github.com:ajdawson/eofs.git'


# ----------------------------------------------------------------------------
# Functions
#
def sh(cmd):
    """Execute command in a subshell, return status code."""
    return check_call(cmd, shell=True)


def sh2(cmd):
    """Execute command in a subshell, return stdout.

    Stderr is unbuffered from the subshell.x.
    
    """
    p = Popen(cmd, stdout=PIPE, shell=True)
    out = p.communicate()[0]
    retcode = p.returncode
    if retcode:
        raise CalledProcessError(retcode, cmd)
    else:
        return out.rstrip()


def init_repo(path):
    """Clone the gh-pages repo to a given location."""
    sh('git clone {} {}'.format(pages_repo, path))
    here = os.getcwdu()
    cd(path)
    sh('git checkout gh-pages')
    cd(here)


# ----------------------------------------------------------------------------
# Script starts
#
if __name__ == '__main__':

    # Determine the commit message that will be used on the branch gh-pages.
    try:
        msg = sys.argv[1]
    except IndexError:
        msg = 'Updated documentation.'
    
    # Check out the repository.
    startdir = os.getcwdu()
    if not os.path.exists(pages_dir):
        # Initialize the repository.
        init_repo(pages_dir)
    else:
        # If the repository exists make sure it is on the right branch and
        # is up-to-date.
        cd(pages_dir)
        sh('git checkout gh-pages')
        sh('git pull')
        cd(startdir)

    # Copy the built documentation to the gh-pages directory.
    sh('rm -rf {}/*'.format(pages_dir))
    sh('cp -r {}/* {}/'.format(html_dir, pages_dir))

    try:

        # Check the correct branch is being used.
        cd(pages_dir)
        status = sh2('git status | head -n 1')
        branch = re.match('On branch (.*)$', status).group(1)
        if branch != 'gh-pages':
            e = 'On {}, git branch is {}, must be "gh-pages"'.format(
                    pages_dir, branch)
            raise RuntimeError(e)

        # Make a commit to the gh-pages branch.
        sh('git add -A')
        sh('git commit -m "{}"'.format(msg))

        # Print a summary of last 3 commits.
        print
        print 'Most recent 3 commits to "gh-pages":'
        sys.stdout.flush()
        sh('git --no-pager log --oneline HEAD~3..')

    finally:

        # Change back to the starting directory.
        cd(startdir)

    # Print a summary message.
    print
    print 'Done. Please verify the build in: {}'.format(pages_dir)
    print 'If everything looks good, "git push origin gh-pages"'

