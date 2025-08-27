// RUN: myproject-opt %s | myproject-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = list.foo %{{.*}} : i32
        %res = list.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @list_types(%arg0: !list.list<i32>)
    func.func @list_types(%arg0: !list.list<i32>) {
        return
    }
}
