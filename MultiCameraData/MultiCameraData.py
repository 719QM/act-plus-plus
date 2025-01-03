# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.1
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

if __import__('os').name == 'nt': import win32api; win32api.LoadLibrary('MultiCameraData.dll')
if __package__ or '.' in __name__:
    from . import _MultiCameraDataWrapper
else:
    import _MultiCameraDataWrapper

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


class SwigPyIterator(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _MultiCameraDataWrapper.delete_SwigPyIterator

    def value(self):
        return _MultiCameraDataWrapper.SwigPyIterator_value(self)

    def incr(self, n=1):
        return _MultiCameraDataWrapper.SwigPyIterator_incr(self, n)

    def decr(self, n=1):
        return _MultiCameraDataWrapper.SwigPyIterator_decr(self, n)

    def distance(self, x):
        return _MultiCameraDataWrapper.SwigPyIterator_distance(self, x)

    def equal(self, x):
        return _MultiCameraDataWrapper.SwigPyIterator_equal(self, x)

    def copy(self):
        return _MultiCameraDataWrapper.SwigPyIterator_copy(self)

    def next(self):
        return _MultiCameraDataWrapper.SwigPyIterator_next(self)

    def __next__(self):
        return _MultiCameraDataWrapper.SwigPyIterator___next__(self)

    def previous(self):
        return _MultiCameraDataWrapper.SwigPyIterator_previous(self)

    def advance(self, n):
        return _MultiCameraDataWrapper.SwigPyIterator_advance(self, n)

    def __eq__(self, x):
        return _MultiCameraDataWrapper.SwigPyIterator___eq__(self, x)

    def __ne__(self, x):
        return _MultiCameraDataWrapper.SwigPyIterator___ne__(self, x)

    def __iadd__(self, n):
        return _MultiCameraDataWrapper.SwigPyIterator___iadd__(self, n)

    def __isub__(self, n):
        return _MultiCameraDataWrapper.SwigPyIterator___isub__(self, n)

    def __add__(self, n):
        return _MultiCameraDataWrapper.SwigPyIterator___add__(self, n)

    def __sub__(self, *args):
        return _MultiCameraDataWrapper.SwigPyIterator___sub__(self, *args)
    def __iter__(self):
        return self

# Register SwigPyIterator in _MultiCameraDataWrapper:
_MultiCameraDataWrapper.SwigPyIterator_swigregister(SwigPyIterator)

FASTCDR_VERSION_MAJOR = _MultiCameraDataWrapper.FASTCDR_VERSION_MAJOR
FASTCDR_VERSION_MINOR = _MultiCameraDataWrapper.FASTCDR_VERSION_MINOR
FASTCDR_VERSION_MICRO = _MultiCameraDataWrapper.FASTCDR_VERSION_MICRO
FASTCDR_VERSION_STR = _MultiCameraDataWrapper.FASTCDR_VERSION_STR
HAVE_CXX11 = _MultiCameraDataWrapper.HAVE_CXX11
FASTCDR_IS_BIG_ENDIAN_TARGET = _MultiCameraDataWrapper.FASTCDR_IS_BIG_ENDIAN_TARGET
FASTCDR_HAVE_FLOAT128 = _MultiCameraDataWrapper.FASTCDR_HAVE_FLOAT128
FASTCDR_SIZEOF_LONG_DOUBLE = _MultiCameraDataWrapper.FASTCDR_SIZEOF_LONG_DOUBLE
import fastdds
class uint8_t_vector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def iterator(self):
        return _MultiCameraDataWrapper.uint8_t_vector_iterator(self)
    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _MultiCameraDataWrapper.uint8_t_vector___nonzero__(self)

    def __bool__(self):
        return _MultiCameraDataWrapper.uint8_t_vector___bool__(self)

    def __len__(self):
        return _MultiCameraDataWrapper.uint8_t_vector___len__(self)

    def __getslice__(self, i, j):
        return _MultiCameraDataWrapper.uint8_t_vector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _MultiCameraDataWrapper.uint8_t_vector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _MultiCameraDataWrapper.uint8_t_vector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _MultiCameraDataWrapper.uint8_t_vector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _MultiCameraDataWrapper.uint8_t_vector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _MultiCameraDataWrapper.uint8_t_vector___setitem__(self, *args)

    def pop(self):
        return _MultiCameraDataWrapper.uint8_t_vector_pop(self)

    def append(self, x):
        return _MultiCameraDataWrapper.uint8_t_vector_append(self, x)

    def empty(self):
        return _MultiCameraDataWrapper.uint8_t_vector_empty(self)

    def size(self):
        return _MultiCameraDataWrapper.uint8_t_vector_size(self)

    def swap(self, v):
        return _MultiCameraDataWrapper.uint8_t_vector_swap(self, v)

    def begin(self):
        return _MultiCameraDataWrapper.uint8_t_vector_begin(self)

    def end(self):
        return _MultiCameraDataWrapper.uint8_t_vector_end(self)

    def rbegin(self):
        return _MultiCameraDataWrapper.uint8_t_vector_rbegin(self)

    def rend(self):
        return _MultiCameraDataWrapper.uint8_t_vector_rend(self)

    def clear(self):
        return _MultiCameraDataWrapper.uint8_t_vector_clear(self)

    def get_allocator(self):
        return _MultiCameraDataWrapper.uint8_t_vector_get_allocator(self)

    def pop_back(self):
        return _MultiCameraDataWrapper.uint8_t_vector_pop_back(self)

    def erase(self, *args):
        return _MultiCameraDataWrapper.uint8_t_vector_erase(self, *args)

    def __init__(self, *args):
        _MultiCameraDataWrapper.uint8_t_vector_swiginit(self, _MultiCameraDataWrapper.new_uint8_t_vector(*args))

    def push_back(self, x):
        return _MultiCameraDataWrapper.uint8_t_vector_push_back(self, x)

    def front(self):
        return _MultiCameraDataWrapper.uint8_t_vector_front(self)

    def back(self):
        return _MultiCameraDataWrapper.uint8_t_vector_back(self)

    def assign(self, n, x):
        return _MultiCameraDataWrapper.uint8_t_vector_assign(self, n, x)

    def resize(self, *args):
        return _MultiCameraDataWrapper.uint8_t_vector_resize(self, *args)

    def insert(self, *args):
        return _MultiCameraDataWrapper.uint8_t_vector_insert(self, *args)

    def reserve(self, n):
        return _MultiCameraDataWrapper.uint8_t_vector_reserve(self, n)

    def capacity(self):
        return _MultiCameraDataWrapper.uint8_t_vector_capacity(self)

    def get_buffer(self):
        return _MultiCameraDataWrapper.uint8_t_vector_get_buffer(self)
    __swig_destroy__ = _MultiCameraDataWrapper.delete_uint8_t_vector

# Register uint8_t_vector in _MultiCameraDataWrapper:
_MultiCameraDataWrapper.uint8_t_vector_swigregister(uint8_t_vector)

class _ImageDataSeq(fastdds.LoanableCollection):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")

    def __init__(self, *args, **kwargs):
        raise AttributeError("No constructor defined - class is abstract")
    __repr__ = _swig_repr
    __swig_destroy__ = _MultiCameraDataWrapper.delete__ImageDataSeq

# Register _ImageDataSeq in _MultiCameraDataWrapper:
_MultiCameraDataWrapper._ImageDataSeq_swigregister(_ImageDataSeq)

class ImageDataSeq(_ImageDataSeq):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    __swig_destroy__ = _MultiCameraDataWrapper.delete_ImageDataSeq

    def __init__(self, *args):
        _MultiCameraDataWrapper.ImageDataSeq_swiginit(self, _MultiCameraDataWrapper.new_ImageDataSeq(*args))

    def __len__(self):
        return _MultiCameraDataWrapper.ImageDataSeq___len__(self)

    def __getitem__(self, i):
        return _MultiCameraDataWrapper.ImageDataSeq___getitem__(self, i)

# Register ImageDataSeq in _MultiCameraDataWrapper:
_MultiCameraDataWrapper.ImageDataSeq_swigregister(ImageDataSeq)

class ImageData(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    __swig_destroy__ = _MultiCameraDataWrapper.delete_ImageData

    def __init__(self, *args):
        _MultiCameraDataWrapper.ImageData_swiginit(self, _MultiCameraDataWrapper.new_ImageData(*args))

    def __eq__(self, x):
        return _MultiCameraDataWrapper.ImageData___eq__(self, x)

    def __ne__(self, x):
        return _MultiCameraDataWrapper.ImageData___ne__(self, x)

    def width(self, *args):
        return _MultiCameraDataWrapper.ImageData_width(self, *args)

    def height(self, *args):
        return _MultiCameraDataWrapper.ImageData_height(self, *args)

    def camera110(self, *args):
        return _MultiCameraDataWrapper.ImageData_camera110(self, *args)

    def camera100(self, *args):
        return _MultiCameraDataWrapper.ImageData_camera100(self, *args)

# Register ImageData in _MultiCameraDataWrapper:
_MultiCameraDataWrapper.ImageData_swigregister(ImageData)

FASTDDS_GEN_API_VER = _MultiCameraDataWrapper.FASTDDS_GEN_API_VER
class ImageDataPubSubType(fastdds.TopicDataType):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr

    def __init__(self):
        _MultiCameraDataWrapper.ImageDataPubSubType_swiginit(self, _MultiCameraDataWrapper.new_ImageDataPubSubType())
    __swig_destroy__ = _MultiCameraDataWrapper.delete_ImageDataPubSubType

    def serialize(self, data, payload, data_representation):
        return _MultiCameraDataWrapper.ImageDataPubSubType_serialize(self, data, payload, data_representation)

    def deserialize(self, payload, data):
        return _MultiCameraDataWrapper.ImageDataPubSubType_deserialize(self, payload, data)

    def calculate_serialized_size(self, data, data_representation):
        return _MultiCameraDataWrapper.ImageDataPubSubType_calculate_serialized_size(self, data, data_representation)

    def compute_key(self, *args):
        return _MultiCameraDataWrapper.ImageDataPubSubType_compute_key(self, *args)

    def create_data(self):
        return _MultiCameraDataWrapper.ImageDataPubSubType_create_data(self)

    def delete_data(self, data):
        return _MultiCameraDataWrapper.ImageDataPubSubType_delete_data(self, data)

    def register_type_object_representation(self):
        return _MultiCameraDataWrapper.ImageDataPubSubType_register_type_object_representation(self)

    def is_bounded(self):
        return _MultiCameraDataWrapper.ImageDataPubSubType_is_bounded(self)

    def is_plain(self, data_representation):
        return _MultiCameraDataWrapper.ImageDataPubSubType_is_plain(self, data_representation)

    def construct_sample(self, memory):
        return _MultiCameraDataWrapper.ImageDataPubSubType_construct_sample(self, memory)

# Register ImageDataPubSubType in _MultiCameraDataWrapper:
_MultiCameraDataWrapper.ImageDataPubSubType_swigregister(ImageDataPubSubType)



