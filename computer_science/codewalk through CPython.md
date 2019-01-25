# CPython Internals

[TOC]

## Overview

Interpreter:

* PyPy: written in python
* Cpython: official one, in C
* Skulpt: in js
* Jython: in java

pyfile - compiler - bytecode - interpreter - output

python 2.7.8, foucs on the interpreter part

main interpreter: ceval.c

`make -jn`, parallel make with n cores

## Opcodes and main interpreter loop

Include/opcode.h, Python/ceval.c

`compile(open('source.py').read(),'name.py','exec')` returns a code object, .co_code gives the bytecode

`python -m dis test.py` something like assembler, implemented by Lib/dis.py, more prefered module: byteplay

valuestack in running, BINARY_op instruction always use the lowest two variables do the operation and pop up the two terms

opcode.h instruction set table to number code

ceval.c: `PyEval_EvalFrameEx()` this is the main interpreter, `**stackpointer` implement the valueslack, giant `for(;;)` main loop and giant `switch(opcode)` in it, VM mimic things in CPU

## Frames, function calls and scope

PY_DECREF, macro for implementation of python garbage recollection

[pythontutor.com] visualization of python code exec

function frames as  function stack

`import dis, test` `dis.dis(test.foo)` explore the opcode within functions

`CALL_FUNCTION` opcode

Include/code.h: bycode object interface

Include/frameobject.h: `PyFrameObject` link list as a frame stack for functions, `*f_back` for parent function stack pos. there is a value stack for each frame obj.

`CALL_FUNCTION`: `call_function` the argument is the number of parameters,`fast_function`

## PyObject

Include/object.h Objects/object.c

should be at least value, type, id, refcount `from sys import getrefcount`

PyObject {type pointer, refcount}  PyIntObject, cast PyObj pointer to behave as a class system

## Python data types

Objects/abstract.c Objects/stringobject.c

`PyVarObject` has variable size

C string implementation, with 0x00 at the end 

Python string, keep a size, imuutable, keep the 0 end, string with null bytes in the middle can be represented

`PyStringObject` with VarObj head, and the `hash` `state`, char `ob_sval`, string  just store in the obj instead of a pointer

`PyObject_RichCompare` in Objects/object.c

## Code objects, function objects, and closures

`fun.func_globals` dict of global variables of functions, function object has global pointer and bytecode pointer as attrs.

Include/code.h: `PyCodeObject`, code obj doesn't contain any global env, and hence cannot run by itself

Objects/codeobject.c:  only CMP in opcode, but try figure out which compare approach to use at runtime.

All code object exists at compiling time. And we link function obj to code obj in runtime.  

`PyFunctionObject` 

closure, nested fun def, with inner variable kept. the nested define fun obj global env pointer is to the parent fun. the search for paramters within a function are searched along the global pointer.`LOAD_DEREF` opcode for closure. `func_closure` for the func's attrs.`func_closure[0].cell_contents`, a mimic of closure behavior.

## Iterators

`i=lst.__iter__()`,`i.next()`. iterator is better for non-trivial organized data structure than index. custom iterator: just implement a `next()` for the class. for in is a shorthand for explicitly iterate of iterator. Opcode for for loop `SETUP_LOOP`,`GET_ITER`,`FOR_ITER`,`JUMP_ABSOLUTE`,`POP_BLOCK`

Objects/iterobject.c 

## User defined classes

Opcode `BUILD_CLASS`, class init opcode is just like an ordinary function call, name, base class, code object for methods. metaclass to create class and be called when class is created.

`PyClassObject`, `PyInstanceObject`: in_class pointer, there is dict in class and dict in instance, `dir` automatically include all attrs from parents' dict. `im_func,im_class,im_self` for method in class `PyMethodObject`

## Generators

Fast way to implement iterators by function with yield within it.

`YIELD_VALUE` opcode. `genobject.h`. PyGenObject, gi_running, guard self call within generator. genobject.c

## Reference

1. http://pgbovine.net/cpython-internals.htm