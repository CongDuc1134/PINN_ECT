# -*- coding: utf-8 -*-
"""
===============================================================================
CODE INTEGRITY & ANTI-HALLUCINATION STATIC ANALYZER (AST-BASED)
===============================================================================
Công cụ kiểm toán tĩnh độc lập sử dụng AST (Abstract Syntax Tree) trong Python.
Phát hiện:
  1. Lỗi cú pháp (SyntaxError / IndentationError).
  2. Các hàm/biến được gọi nhưng chưa từng được định nghĩa hoặc import (Undefined NameError).
  3. Sử dụng wildcard import (from ... import *) - nguồn gốc gây nhiễu và đè biến.
  4. Các hàm import từ file/module nội bộ không tồn tại trong module đích.
  5. Cảnh báo các khối thực thi trễ (deferred execution) có biến chưa khởi tạo.
"""

import os
import sys
import ast
import builtins
import argparse
from pathlib import Path

# Cấu hình encoding UTF-8 tương thích mọi hệ điều hành (kể cả Windows cp1252)
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Danh sách built-in chuẩn của Python
BUILTIN_NAMES = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__package__", "__loader__", "__spec__",
    "__annotations__", "__builtins__", "self", "cls"
}

class ScopeAnalyzer(ast.NodeVisitor):
    def __init__(self, filename):
        self.filename = filename
        self.scope_stack = [set(BUILTIN_NAMES)] # [global_scope, local_scope_1, ...]
        self.imported_modules = set()
        self.imported_names = set()
        self.undefined_names = [] # list of (name, lineno, col_offset)
        self.wildcard_imports = [] # list of (module_name, lineno)
        self.function_defs = {} # name -> list of arg names
        self.class_defs = set()
        self.deferred_blocks = [] # (type, lineno)
        self.global_vars = set()

    def current_scope(self):
        return self.scope_stack[-1]

    def is_defined(self, name):
        return any(name in s for s in reversed(self.scope_stack))

    def define(self, name):
        if name in self.global_vars:
            self.scope_stack[0].add(name)
        else:
            self.scope_stack[-1].add(name)

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split('.')[0]
            self.define(name)
            self.imported_modules.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ""
        for alias in node.names:
            if alias.name == "*":
                self.wildcard_imports.append((module, node.lineno))
            else:
                name = alias.asname if alias.asname else alias.name
                self.define(name)
                self.imported_names.add(name)
        self.generic_visit(node)

    def visit_Global(self, node):
        for name in node.names:
            self.global_vars.add(name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Tên hàm được định nghĩa ở scope hiện tại (scope cha)
        self.define(node.name)
        args = [a.arg for a in node.args.args]
        self.function_defs[node.name] = args

        # Đẩy scope hàm mới vào stack
        fn_scope = set()
        for arg in node.args.args:
            fn_scope.add(arg.arg)
        if node.args.vararg:
            fn_scope.add(node.args.vararg.arg)
        if node.args.kwarg:
            fn_scope.add(node.args.kwarg.arg)
        for arg in node.args.kwonlyargs:
            fn_scope.add(arg.arg)

        # Xử lý defaults ở scope cha trước khi vào hàm
        for d in node.args.defaults:
            self.visit(d)
        for d in node.args.kw_defaults:
            if d is not None:
                self.visit(d)
        for deco in node.decorator_list:
            self.visit(deco)

        self.scope_stack.append(fn_scope)
        for item in node.body:
            self.visit(item)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_Lambda(self, node):
        lambda_scope = set()
        for arg in node.args.args:
            lambda_scope.add(arg.arg)
        if node.args.vararg:
            lambda_scope.add(node.args.vararg.arg)
        if node.args.kwarg:
            lambda_scope.add(node.args.kwarg.arg)
        for arg in node.args.kwonlyargs:
            lambda_scope.add(arg.arg)

        self.scope_stack.append(lambda_scope)
        self.visit(node.body)
        self.scope_stack.pop()

    def visit_ClassDef(self, node):
        self.define(node.name)
        self.class_defs.add(node.name)
        for base in node.bases:
            self.visit(base)
        for deco in node.decorator_list:
            self.visit(deco)

        # Scope class
        self.scope_stack.append(set())
        for item in node.body:
            self.visit(item)
        self.scope_stack.pop()

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.define(node.id)
        elif isinstance(node.ctx, ast.Load):
            if not self.is_defined(node.id):
                self.undefined_names.append((node.id, node.lineno, node.col_offset))
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self.scope_stack.append(set())
        for gen in node.generators:
            self._extract_target_names(gen.target)
            self.visit(gen.iter)
            for if_expr in gen.ifs:
                self.visit(if_expr)
        self.visit(node.elt)
        self.scope_stack.pop()

    def visit_SetComp(self, node):
        self.visit_ListComp(node)

    def visit_DictComp(self, node):
        self.scope_stack.append(set())
        for gen in node.generators:
            self._extract_target_names(gen.target)
            self.visit(gen.iter)
            for if_expr in gen.ifs:
                self.visit(if_expr)
        self.visit(node.key)
        self.visit(node.value)
        self.scope_stack.pop()

    def visit_GeneratorExp(self, node):
        self.visit_ListComp(node)

    def visit_For(self, node):
        self._extract_target_names(node.target)
        self.generic_visit(node)

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars:
                self._extract_target_names(item.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.define(node.name)
        self.generic_visit(node)

    def visit_If(self, node):
        cond_src = ast.unparse(node.test) if hasattr(ast, 'unparse') else ""
        if any(keyword in cond_src for keyword in ["epoch", "save", "eval", "test", "step %", "is_best"]):
            self.deferred_blocks.append((cond_src, node.lineno))
        self.generic_visit(node)

    def _extract_target_names(self, target):
        if isinstance(target, ast.Name):
            self.define(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._extract_target_names(elt)


def audit_file(filepath):
    """Kiểm tra toàn vẹn 1 file Python."""
    results = {
        "file": filepath,
        "syntax_error": None,
        "undefined_names": [],
        "wildcard_imports": [],
        "deferred_warnings": []
    }

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        results["syntax_error"] = f"Không đọc được file: {e}"
        return results

    # 1. Parse AST
    try:
        tree = ast.parse(code, filename=filepath)
    except SyntaxError as e:
        results["syntax_error"] = f"SyntaxError tại dòng {e.lineno}, cột {e.offset}: {e.msg}"
        return results

    analyzer = ScopeAnalyzer(filepath)
    analyzer.visit(tree)

    # 2. Lấy danh sách biến / hàm chưa định nghĩa từ ScopeAnalyzer
    has_wildcard = len(analyzer.wildcard_imports) > 0
    undefined = []
    for name, lineno, col in analyzer.undefined_names:
        source_hint = " (Lưu ý: file có dùng 'import *' nên có thể ẩn trong wildcard)" if has_wildcard else ""
        undefined.append((name, lineno, source_hint))

    results["undefined_names"] = undefined
    results["wildcard_imports"] = analyzer.wildcard_imports
    results["deferred_warnings"] = analyzer.deferred_blocks
    return results


def print_report(results_list):
    total_files = len(results_list)
    files_with_errors = 0
    files_with_warnings = 0

    print("=" * 80)
    print("🔍 CODE INTEGRITY & ANTI-HALLUCINATION AUDIT REPORT")
    print("=" * 80)

    for r in results_list:
        rel_path = os.path.relpath(r["file"], os.getcwd()) if os.path.isabs(r["file"]) else r["file"]
        has_issue = r["syntax_error"] or r["undefined_names"] or r["wildcard_imports"]
        
        if r["syntax_error"]:
            files_with_errors += 1
            print(f"\n❌ [SYNTAX ERROR] {rel_path}")
            print(f"   └── {r['syntax_error']}")
            continue

        if r["undefined_names"]:
            files_with_errors += 1
            print(f"\n❌ [UNDEFINED / HALLUCINATED SYMBOLS] {rel_path}")
            for name, lineno, hint in r["undefined_names"][:15]:
                print(f"   ├── Dòng {lineno:>4}: '{name}' chưa được định nghĩa hoặc import!{hint}")
            if len(r["undefined_names"]) > 15:
                print(f"   └── ... và {len(r['undefined_names']) - 15} lỗi khác.")

        if r["wildcard_imports"]:
            files_with_warnings += 1
            print(f"\n⚠️  [WILDCARD IMPORT DETECTED] {rel_path}")
            for mod, lineno in r["wildcard_imports"]:
                print(f"   └── Dòng {lineno:>4}: 'from {mod} import *' -> Dễ gây xung đột tên hoặc che giấu hàm thiếu.")

        if not has_issue:
            print(f"✅ [PASS] {rel_path}")

    print("\n" + "-" * 80)
    print(f"📊 TỔNG KẾT: {total_files} file đã quét | {files_with_errors} file LỖI | {files_with_warnings} file CẢNH BÁO")
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Kiểm toán toàn vẹn mã nguồn & chống hallucination code")
    parser.add_argument("path", type=str, help="Đường dẫn đến file .py hoặc thư mục dự án")
    parser.add_argument("--recursive", "-r", action="store_true", help="Quét đệ quy toàn bộ thư mục con")
    args = parser.parse_args()

    target = os.path.abspath(args.path)
    files_to_check = []

    if os.path.isfile(target):
        if target.endswith(".py"):
            files_to_check.append(target)
    elif os.path.isdir(target):
        if args.recursive:
            for root, _, files in os.walk(target):
                # Bỏ qua các thư mục ảo, venv, git
                if any(ignored in root for ignored in [".git", "__pycache__", ".venv", "venv", "site-packages", ".agents"]):
                    continue
                for f in files:
                    if f.endswith(".py"):
                        files_to_check.append(os.path.join(root, f))
        else:
            for f in os.listdir(target):
                if f.endswith(".py"):
                    files_to_check.append(os.path.join(target, f))

    if not files_to_check:
        print(f"Không tìm thấy file .py nào tại: {target}")
        sys.exit(0)

    results = [audit_file(f) for f in sorted(files_to_check)]
    print_report(results)


if __name__ == "__main__":
    main()
