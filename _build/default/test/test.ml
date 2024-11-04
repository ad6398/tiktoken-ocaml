(* File: test_ocaml_rust_tiktok.ml *)

(* The Rust function should be available via FFI and accessible from the Ocaml_rust_tiktok module *)
let test_core_bpe_new () =
  (* Create a sample encoder list, special tokens encoder list, and a pattern *)
  let encoder = [(Bytes.of_string "example", 1); (Bytes.of_string "test", 2)] in
  let special_tokens_encoder = [("token1", 1); ("token2", 2)] in
  let pattern = "pattern" in

  (* Call the Rust function from the Ocaml_rust_tiktok module *)
  let id = Ocaml_rust_tiktok.core_bpe_new encoder special_tokens_encoder pattern in

  (* Print the returned id to verify the result *)
  Printf.printf "Returned core BPE id: %d\n" id

(* Run the test *)
let () = test_core_bpe_new ()
 
