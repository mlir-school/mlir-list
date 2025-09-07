// RUN: listproject-opt %s | listproject-opt | FileCheck %s

module {
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  // CHECK: %{{.*}} = list.range %{{.*}} to %{{.*}} : !list.list<i32>
  %int_list = list.range %c0 to %c5 : !list.list<i32>
  // CHECK: %{{.*}} = list.length %{{.*}} : !list.list<i32> -> i32
  %length = list.length %int_list : !list.list<i32> -> i32
  // CHECK: %{{.*}} = list.map %{{.*}} with (%{{.*}} : i32) -> i1 {
  %0 = list.map %int_list with (%x : i32) -> i1 {
    %c2 = arith.constant 2 : i32
    %x_ult_c2 = arith.cmpi ult, %x, %c2 : i32
    // CHECK: list.yield  %{{.*}} : i1
    list.yield %x_ult_c2 : i1
  }
}
