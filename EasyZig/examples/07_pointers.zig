// Chapter 7: Pointers
// Memory addresses and indirection

const std = @import("std");

pub fn main() void {
    single_item_pointers();
    many_item_pointers();
    pointer_arithmetic();
    const_pointers();
}

fn single_item_pointers() void {
    std.debug.print("=== Single-Item Pointers ===\n", .{});

    // Create a variable
    var value: i32 = 42;

    // Get pointer to the variable
    const ptr: *i32 = &value;

    std.debug.print("value = {d}\n", .{value});
    std.debug.print("ptr.* = {d}\n", .{ptr.*}); // Dereference

    // Modify through pointer
    ptr.* = 100;
    std.debug.print("After ptr.* = 100:\n", .{});
    std.debug.print("value = {d}\n", .{value});

    // Pointer to pointer
    var x: i32 = 5;
    const p1: *i32 = &x;
    const p2: **i32 = &p1;

    std.debug.print("x = {d}, *p1 = {d}, **p2 = {d}\n", .{ x, p1.*, p2.*.* });
}

fn many_item_pointers() void {
    std.debug.print("\n=== Many-Item Pointers ===\n", .{});

    var array = [_]i32{ 10, 20, 30, 40, 50 };

    // Many-item pointer (unknown length)
    const ptr: [*]i32 = &array;

    // Access elements via indexing
    std.debug.print("ptr[0] = {d}\n", .{ptr[0]});
    std.debug.print("ptr[2] = {d}\n", .{ptr[2]});
    std.debug.print("ptr[4] = {d}\n", .{ptr[4]});

    // Convert to slice for safe iteration
    const slice = ptr[0..5];
    std.debug.print("As slice: {any}\n", .{slice});
}

fn pointer_arithmetic() void {
    std.debug.print("\n=== Pointer Arithmetic ===\n", .{});

    var array = [_]i32{ 100, 200, 300, 400, 500 };
    const ptr: [*]i32 = &array;

    // Pointer addition
    const ptr_plus_2 = ptr + 2;
    std.debug.print("ptr[0] = {d}\n", .{ptr[0]});
    std.debug.print("(ptr + 2)[0] = {d}\n", .{ptr_plus_2[0]});

    // Pointer subtraction
    const ptr_end: [*]i32 = ptr + 5;
    const diff = @intFromPtr(ptr_end) - @intFromPtr(ptr);
    std.debug.print("Distance in bytes: {d}\n", .{diff});
}

fn const_pointers() void {
    std.debug.print("\n=== Const Pointers ===\n", .{});

    var mutable: i32 = 10;
    const immutable: i32 = 20;

    // Pointer to mutable data
    const ptr_mut: *i32 = &mutable;
    ptr_mut.* = 15;
    std.debug.print("Modified mutable: {d}\n", .{mutable});

    // Pointer to const data (cannot modify)
    const ptr_const: *const i32 = &immutable;
    std.debug.print("Const value: {d}\n", .{ptr_const.*});
    // ptr_const.* = 25; // Error: cannot modify const

    // Mutable pointer can be cast to const
    const ptr_as_const: *const i32 = &mutable;
    std.debug.print("Mutable as const: {d}\n", .{ptr_as_const.*});
}

// Passing by pointer
fn increment(ptr: *i32) void {
    ptr.* += 1;
}

fn double_values(values: []i32) void {
    for (values) |*v| {
        v.* *= 2;
    }
}

pub fn pointer_parameters() void {
    std.debug.print("\n=== Pointer Parameters ===\n", .{});

    var num: i32 = 5;
    std.debug.print("Before increment: {d}\n", .{num});
    increment(&num);
    std.debug.print("After increment: {d}\n", .{num});

    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    std.debug.print("Before double: {any}\n", .{arr});
    double_values(&arr);
    std.debug.print("After double: {any}\n", .{arr});
}

// Optional pointers (nullable)
pub fn optional_pointers() void {
    std.debug.print("\n=== Optional Pointers ===\n", .{});

    var value: i32 = 42;
    var opt_ptr: ?*i32 = &value;

    // Check and unwrap
    if (opt_ptr) |ptr| {
        std.debug.print("Pointer value: {d}\n", .{ptr.*});
    }

    // Set to null
    opt_ptr = null;

    if (opt_ptr) |ptr| {
        std.debug.print("Value: {d}\n", .{ptr.*});
    } else {
        std.debug.print("Pointer is null\n", .{});
    }
}
