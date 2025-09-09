// RUN: listproject-opt --list-remove-some-foo %s | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK-NOT: %{{.*}} = list.foo %{{.*}} {useless} : i32
	    %1 = list.foo %0 {useless} : i32 
        // CHECK: %{{.*}} = list.foo %{{.*}} {not_useless} : i32
        %res = list.foo %1 {not_useless} : i32 
        return
    }
}
