// Chapter 6: Slices
// Views into arrays and memory

const std = @import("std");

pub fn main() void {
    basic_slices();
    slice_operations();
    slice_from_pointer();
}

fn basic_slices() void {
    std.debug.print("=== Basic Slices ===\n", .{});

    // Create an array
    const array = [_]i32{ 10, 20, 30, 40, 50, 60, 70 };

    // Create a slice from array
    const slice: []const i32 = &array;
    std.debug.print("Full slice: {any}\n", .{slice});

    // Slice with range (start..end)
    const middle = array[2..5]; // Elements at index 2, 3, 4
    std.debug.print("middle[2..5]: {any}\n", .{middle});

    // Slice from start
    const first_three = array[0..3];
    std.debug.print("first_three[0..3]: {any}\n", .{first_three});

    // Slice to end
    const last_four = array[3..];
    std.debug.print("last_four[3..]: {any}\n", .{last_four});

    // Open-ended slice (entire array)
    const all = array[0..];
    std.debug.print("all[0..]: {any}\n", .{all});

    // Slice length and access
    std.debug.print("middle.len = {d}\n", .{middle.len});
    std.debug.print("middle[0] = {d}\n", .{middle[0]});
}

fn slice_operations() void {
    std.debug.print("\n=== Slice Operations ===\n", .{});

    // Mutable slice
    var array = [_]i32{ 1, 2, 3, 4, 5 };
    const slice: []i32 = &array;

    // Modify through slice
    slice[0] = 100;
    slice[4] = 500;
    std.debug.print("Modified via slice: {any}\n", .{slice});

    // Iterate over slice
    std.debug.print("Iteration: ", .{});
    for (slice) |value| {
        std.debug.print("{d} ", .{value});
    }
    std.debug.print("\n", .{});

    // Iterate with index
    for (slice, 0..) |value, i| {
        std.debug.print("slice[{d}] = {d}\n", .{ i, value });
    }

    // Pointer to element
    const ptr = &slice[2];
    ptr.* = 999;
    std.debug.print("After pointer modify: {any}\n", .{slice});
}

fn slice_from_pointer() void {
    std.debug.print("\n=== Slice from Pointer ===\n", .{});

    var array = [_]u8{ 'H', 'e', 'l', 'l', 'o', '!', 0 };

    // Many-item pointer
    const ptr: [*]u8 = &array;

    // Convert pointer to slice with known length
    const slice = ptr[0..5];
    std.debug.print("Slice from pointer: {s}\n", .{slice});

    // Sentinel-terminated slice
    const sentinel_slice: [:0]u8 = array[0..6 :0];
    std.debug.print("Sentinel slice: {s}\n", .{sentinel_slice});
}

// Function that takes a slice (flexible)
fn sum(values: []const i32) i32 {
    var total: i32 = 0;
    for (values) |v| {
        total += v;
    }
    return total;
}

fn print_slice(data: []const u8) void {
    std.debug.print("Data: {s}\n", .{data});
}

pub fn slice_as_parameter() void {
    std.debug.print("\n=== Slice as Parameter ===\n", .{});

    const arr1 = [_]i32{ 1, 2, 3 };
    const arr2 = [_]i32{ 10, 20, 30, 40, 50 };

    // Same function works with different sized arrays
    std.debug.print("Sum of arr1: {d}\n", .{sum(&arr1)});
    std.debug.print("Sum of arr2: {d}\n", .{sum(&arr2)});

    // Works with partial slices too
    std.debug.print("Sum of arr2[1..4]: {d}\n", .{sum(arr2[1..4])});

    // String slice
    print_slice("Hello, Zig!");
}
