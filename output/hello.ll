; ModuleID = 'module'
source_filename = "module"

define i32 @sum(i32 %0, i32 %1) {
entry:
  %int_add = add i32 %0, %1
  ret i32 %int_add
}

define i32 @main() {
entry:
  ret i32 3
}
