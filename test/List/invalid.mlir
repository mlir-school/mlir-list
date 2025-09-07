// RUN: listproject-opt -split-input-file %s -verify-diagnostics

module {
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  %int_list = list.range %c0 to %c5 : !list.list<i32>
  // expected-error @+1 {{Element type of the result list does not match the type of the yielded value: (}}
  %0 = list.map %int_list with (%x : i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %x_ult_c2 = arith.cmpi ult, %x, %c2 : i32
    list.yield %x_ult_c2 : i1
  }
}
