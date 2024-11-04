# tiktoken-ocaml
This is my noob attempt to create an Ocaml version of original tiktoken library. I started with changing original rust binders written for python to be called from a OCaml program. I could not write a reasonable data type converter(b/w Rust and Ocaml) for original 'CoreBPE' class/object so instead of exchanging CoreBPE objects they exchange an ID( `int`) which binds to  a specific CoreBPE object. Rust creates these objects and store in the Key-Value with IDs as key.

Inspired from this project using [ocaml-rs](https://github.com/zshipko/ocaml-rs) to call Rust functions from OCaml.


## Building

    dune build

to run the tests:

    dune runtest

to load your library into an interactive sesssion:

  OCAML_INTEROP_NO_CAML_STARTUP=1 dune utop

The `OCAML_INTEROP_NO_CAML_STARTUP` environment variable should be set to ensure
the library is linked correctly.

