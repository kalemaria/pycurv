"""
Set of classes for handling exceptions and warnings implemented for PySeg
package and also used in the PySurf package.

Author: Antonio Martinez-Sanchez (Max Planck Institute for Biochemistry), date:
2014-04-02
"""

__author__ = 'martinez'
__version__ = "$Revision: 001 $"


# Exceptions

class PySegError(Exception):
    """Base class for pexceptions in this module."""

    pass


class PySegInputError(PySegError):
    """
    Exception raised for errors in the input.

    The constructor requires the following parameters.

    Args:
        expr (str): input expression in which the error occurred
        msg (str): explanation of the error
    """

    def __init__(self, expr, msg):
        """
        Constructor.

        Args:
            expr (str): input expression in which the error occurred
            msg (str): explanation of the error

        Returns:
            None
        """
        self.expr = expr
        self.msg = msg

    def get_message(self):
        """
        Gets the input expression and the message of the error.

        Returns:
            string in format '<expr> - <msg>'
        """
        return self.expr + ' - ' + self.msg


class PySegTransitionError(PySegError):
    """
    Raised when an operation attempts a state transition that is not allowed.

    The constructor requires the following parameters.

    Args:
        prev: state at beginning of transition
        next: attempted new state
        msg (str): explanation of why the specific transition is not allowed
    """

    def __init__(self, prev, next, msg):
        """
        Constructor.

        Args:
            prev: state at beginning of transition
            next: attempted new state
            msg (str): explanation of why the specific transition is not allowed

        Returns:
            None
        """
        self.prev = prev
        self.next = next
        self.msg = msg


# Warnings

class PySegWarning(Warning):
    """Base class for warnings pexceptions in this module."""

    pass


class PySegInputWarning(PySegWarning):
    """
    Exception raised for warnings in the input.

    The constructor requires the following parameters.

    Args:
        expr (str): input expression in which the warning occurred
        msg (str): explanation of the warning
    """

    def __init__(self, expr, msg):
        """
        Constructor.

        Args:
            expr (str): input expression in which the warning occurred
            msg (str): explanation of the warning

        Returns:
            None
        """
        self.expr = expr
        self.msg = msg

    def get_message(self):
        """
        Gets the input expression and the message of the warning.

        Returns:
            string in format '<expr> - <msg>'
        """
        return self.expr + ' - ' + self.msg
