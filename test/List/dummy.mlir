// RUN: listproject-opt %s | listproject-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = list.foo %{{.*}} : i32
        %res = list.foo %0 : i32
        return
    }
}

