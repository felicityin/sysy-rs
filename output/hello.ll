; ModuleID = 'module'
source_filename = "module"

define i32 @sum(i32 %a1, i32 %b2) {
entry:
  %int_add = add i32 %a1, %b2
  ret i32 %int_add
}

define i32 @main() {
entry:
  ret i32 3
}
