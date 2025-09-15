import os
from win32com.client import makepy
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    import GeneratedMcStream
    from GeneratedMcStream import MCSSTRM
else:
    # this need to be run first to enable IDE autocomplete
    # auto complete in ipython is working directly after import of MCStream
    try:
        import GeneratedMcStream
    except ImportError:
        file_name = 'GeneratedMcStream.py'
        if not os.path.exists(file_name):
            with open(file_name, "w") as f:
                makepy.GenerateFromTypeLibSpec("MCStream", file=f)
        import GeneratedMcStream


def get_generated_mc_stream():
    return GeneratedMcStream


if TYPE_CHECKING:
    # class for IDE to enable autocomplete
    class MCStreamWrapper(MCSSTRM.default_interface):
        def __init__(self, oobj=...):
            super().__init__(oobj)
else:
    class MCStreamWrapper():
        def __init__(self):
            self._module = get_generated_mc_stream()
            self._mcstream = self._module.MCSSTRM()
            self._stream = self._mcstream.default_interface

        def get_com(self):
            return self._module

        def __getattr__(self, name):
            """
            Forward method/property to the stream interface.
            """
            attr = getattr(self._stream, name)
            if callable(attr):
                def wrapped_method(*args, **kwargs):
                    return attr(self._mcstream, *args, **kwargs)

                return wrapped_method

            return attr

        def __dir__(self):
            """
            Makes IPython and dir() show all members of the default_interface.
            """
            return sorted(set(
                dir(type(self)) +  # own class methods
                dir(self._stream)  # methods from the COM interface
            ))


    T1 = TypeVar('T1')


    class MCSObject2ClassWrapper(Generic[T1]):
        def __init__(self, class_definiton, obj):

            if None == obj:
                msg = "None, cannot be instance of {0}".format(class_definiton)
                raise TypeError(msg)


            self._module = get_generated_mc_stream()
            self._mcs_object = obj


            valid_name = [item for item in dir(self._module) if item.lower() == class_definiton.__name__.lower()]
            if len(valid_name) == 0:
                raise ValueError(f"Interface for {class_definiton.__name__} not found in module.")

            iface_cls = getattr(self._module, valid_name[0], None)
            self._interface = iface_cls

        def __getattr__(self, name):
            """
            Forward method/property to the stream interface.
            """
            attr = getattr(self._interface, name)
            if callable(attr):
                def wrapped_method(*args, **kwargs):
                    return attr(self._mcs_object, *args, **kwargs)

                return wrapped_method

            return attr

        def __dir__(self):
            """
            Makes IPython and dir() show all members of the default_interface.
            """
            return sorted(set(
                dir(type(self)) +  # own class methods
                dir(self._interface)  # methods from the COM interface
            ))
