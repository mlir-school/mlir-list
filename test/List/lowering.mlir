// RUN: listproject-opt --lower-list %s | FileCheck %s

module {
  %c0 = arith.constant 0 : i32
  %c5 = arith.constant 5 : i32
  %int_list = list.range %c0 to %c5 : !list.list<i32>
  %0 = list.map %int_list with (%x : i32) -> i1 {
    %c2 = arith.constant 2 : i32
    %x_ult_c2 = arith.cmpi ult, %x, %c2 : i32
    list.yield %x_ult_c2 : i1
  }
}

// CHECK:  %{{.*}} = tensor.empty(%{{.*}}) : tensor<?xi32>
// CHECK:  %{{.*}} = scf.for %arg0 = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<?xi32>) {
// CHECK:    %{{.*}} = arith.index_cast %{{.*}} : index to i32
// CHECK:    %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
// CHECK:    %{{.*}} = tensor.insert %{{.*}} into %{{.*}}[%{{.*}}] : tensor<?xi32>
// CHECK:    scf.yield %{{.*}} : tensor<?xi32>
// CHECK:  }
// CHECK:  %{{.*}} = tensor.dim %{{.*}}, %{{.*}} : tensor<?xi32>
// CHECK:  %{{.*}} = tensor.empty(%{{.*}}) : tensor<?xi1>
// CHECK:  %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<?xi1>) {
// CHECK:    %{{.*}} = tensor.extract %{{.*}}[%{{.*}}] : tensor<?xi32>
// CHECK:    %{{.*}} = arith.cmpi ult, %{{.*}}, %{{.*}} : i32
// CHECK:    %{{.*}} = tensor.insert %{{.*}} into %{{.*}}[%{{.*}}] : tensor<?xi1>
// CHECK:    scf.yield %{{.*}} : tensor<?xi1>
// CHECK:  }
// CHECK:}
