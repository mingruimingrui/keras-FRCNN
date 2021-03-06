import sys
import inspect
import textwrap
from collections import OrderedDict

from .collections import AttrDict
from ._validation import (
    is_int_like,
    is_float_like,
    is_list_like
)


class ConfigTemplate:
    """ For help on ConfigTemplate, use ConfigTemplate(help=True) """

    def __init__(self, help=False, **kwargs):
        self.__name__ = 'ConfigTemplate'
        self.__params__ = OrderedDict()

        """Place all your self.add here and remove Error"""
        raise NotImplementedError('__init__ not implemented')

        if help:
            self.help()
        else:
            self._validate_kwargs_(**kwargs)


    def add(self, name, desc, **kwargs):
        """ Adds a new condition to the config
        other than name and desc, the rest are optional

        Args
            name           - Name of your param (str)
            desc           - Description for your param (str)
            default        - Default value for your param (default None)
            accepted_types - Tuple (or single variable) of accepted types for your param
            valid_options  - List-like of accepted entries for your param
            condition      - Required condition for your param (func)
            required       - Value of param cannot be None (bool)

        """
        self.__params__[name] = {'desc': desc}

        for kwarg in kwargs:
            assert kwarg in ['default', 'accepted_types', 'valid_options', 'condition', 'required'], \
                '{} is not a valid add kwarg'.format(kwarg)

        for kwarg in ['default', 'accepted_types', 'valid_options', 'condition', 'required']:
            if kwarg in kwargs:
                self.__params__[name][kwarg] = kwargs[kwarg]
            else:
                self.__params__[name][kwarg] = None


    def _validate_kwargs_(self, **kwargs):
        try:
            self._validate_kwargs(**kwargs)
        except Exception as e:
            self.help()
            sys.exit('AssertionError: ' + str(e))


    def _validate_kwargs(self, **kwargs):
        for param_name, param_reqs in self.__params__.items():
            # Ensure that parameter name is valid
            assert not hasattr(self, param_name), \
                '{} is a repeated param or there might be some problems with the code'.format(param_name) + \
                '\nplease contact the developer'

            # Check if user provided value for parameter
            if param_name in kwargs:
                param_val = kwargs[param_name]
            else:
                param_val = param_reqs['default']

            # Check if required
            if param_reqs['required']:
                assert param_val is not None, '{} is a required field'.format(param_name)
            else:
                if param_val is None:
                    setattr(self, param_name, param_val)
                    continue

            # Check if accpted type
            if param_reqs['accepted_types'] is not None:
                assert check_accpted_types(param_reqs['accepted_types'], param_val), \
                    '{} is not of accpted_types {}'.format(param_name, param_reqs['accepted_types'])

            # Check if in valid options
            if param_reqs['valid_options'] is not None:
                assert param_val in param_reqs['valid_options'], \
                    '{} is not in valid_options {}'.format(param_name, param_reqs['valid_options'])

            # Check if user defined condition is satisfied
            if param_reqs['condition'] is not None:
                assert param_reqs['condition'](param_val), \
                    '{} does not satisfy condition'.format(param_name)

            setattr(self, param_name, param_val)


    def help(self):
        print('*** {} Parameter Guide ***'.format(self.__name__))
        print('Here are a list of parameters and requirements\n')

        for param_name, param_reqs in self.__params__.items():
            if param_reqs['required']:
                print('\n    ==== Param_name - {} (required) ===='.format(param_name))
            else:
                print('\n    ==== Param_name - {} ===='.format(param_name))

            # print('    Description     - {}'.format(param_reqs['desc']))
            print_description(param_reqs['desc'])
            print('    Default         - {}'.format(param_reqs['default']))

            if param_reqs['accepted_types'] is not None:
                print('    Accepted types  - {}'.format(param_reqs['accepted_types']))

            if param_reqs['valid_options'] is not None:
                print('    Valid options   - {}'.format(param_reqs['valid_options']))

            if param_reqs['condition']:
                print('    Condition       -')
                # print(inspect.getsource(param_reqs['condition']))
                print_condition(param_reqs['condition'])
            else:
                print('')

        print('\n*** End of Parameter Guide ***')


    def as_attr_dict(self):
        # Determine attributes to be transfered into AttrDict
        attributes = [attr for attr in dir(self) if
            not attr.startswith('_') and
            attr not in ['add', 'help', 'as_attr_dict']]

        # Create a copy of self as an AttrDict
        copy_self = AttrDict()
        for attr in attributes:
            setattr(copy_self, attr, getattr(self, attr))

        # Enforce self to be immutable
        copy_self.immutable(True)

        return copy_self


def print_description(desc):
    template    = '    Description     - {}'
    indentation = '                      '

    desc = template.format(desc)
    lines = desc.split('\n')
    new_lines = []

    for l in lines:
        if len(l) > 100:
            l = textwrap.fill(l, width=100, subsequent_indent=indentation)

        new_lines.append(l)

    print(('\n' + indentation).join(new_lines))



def print_condition(condition):
    cstr = inspect.getsource(condition)
    len_lws = len(cstr) - len(cstr.lstrip(' '))

    cstr = [s[len_lws:] for s in cstr.split('\n')]
    cstr = ('\n' + ' ' * 8).join(cstr)

    print(' ' * 8 + cstr)


def check_accpted_types(accepted_types, param_val):
    if not is_list_like(accepted_types):
        accepted_types = [accepted_types]

    if isinstance(param_val,
        tuple(at for at in accepted_types if isinstance(at, type))):
        return True

    for at in (at for at in accepted_types if not isinstance(at, type)):
        if at == 'numeric':
            if is_int_like(param_val) or is_float_like(param_val):
                return True
        if at == 'int-like':
            if is_int_like(param_val):
                return True
        elif at == 'float-like':
            if is_float_like(param_val):
                return True
        elif at == 'list-like':
            if is_list_like(param_val):
                return True
        else:
            raise ValueError('Invalid accepted_type received: {}'.format(at))

    return False
