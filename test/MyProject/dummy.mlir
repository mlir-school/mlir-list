// RUN: myproject-opt %s | myproject-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = mydialect.foo %{{.*}} : i32
        %res = mydialect.foo %0 : i32
        return
    }

    // CHECK-LABEL: func @mydialect_types(%arg0: !mydialect.custom<"10">)
    func.func @mydialect_types(%arg0: !mydialect.custom<"10">) {
        return
    }
}
