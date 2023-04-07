; ModuleID = 'module'
source_filename = "module"

define i32 @main() {
entry:
  br i32 2, label %if_block, label %else_block

if_block:                                         ; preds = %entry
  %int_add = add i32 2, 1
  br label %after_block

else_block:                                       ; preds = %entry
  br label %after_block

after_block:                                      ; preds = %else_block, %if_block
  %a.0 = phi i32 [ %int_add, %if_block ], [ 0, %else_block ]
  ret i32 %a.0
}
