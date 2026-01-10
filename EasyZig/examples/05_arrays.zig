// Chapter 5: Arrays
// Fixed-size collections in Zig

const std = @import("std");

pub fn main() void {
    basic_arrays();
    array_operations();
    multidimensional_arrays();
    sentinel_arrays();
}

fn basic_arrays() void {
    std.debug.print("=== Basic Arrays ===\n", .{});

    // Array declaration with explicit type
    const numbers: [5]i32 = [5]i32{ 1, 2, 3, 4, 5 };

    // Array with inferred length
    const fruits = [_][]const u8{ "apple", "banana", "cherry" };

    // Accessing elements
    std.debug.print("First number: {d}\n", .{numbers[0]});
    std.debug.print("Last number: {d}\n", .{numbers[4]});
    std.debug.print("First fruit: {s}\n", .{fruits[0]});

    // Array length
    std.debug.print("Numbers length: {d}\n", .{numbers.len});
    std.debug.print("Fruits length: {d}\n", .{fruits.len});

    // Initialize with same value
    const zeros = [_]i32{0} ** 10; // [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    std.debug.print("Zeros: {any}\n", .{zeros});

    // Initialize with pattern
    const pattern = [_]i32{ 1, 2 } ** 3; // [1, 2, 1, 2, 1, 2]
    std.debug.print("Pattern: {any}\n", .{pattern});
}

fn array_operations() void {
    std.debug.print("\n=== Array Operations ===\n", .{});

    // Modifying array elements (must be var)
    var mutable = [_]i32{ 1, 2, 3, 4, 5 };
    mutable[0] = 100;
    mutable[4] = 500;
    std.debug.print("Modified: {any}\n", .{mutable});

    // Iterating over arrays
    std.debug.print("Iteration: ", .{});
    for (mutable) |value| {
        std.debug.print("{d} ", .{value});
    }
    std.debug.print("\n", .{});

    // Iteration with index
    std.debug.print("With index: ", .{});
    for (mutable, 0..) |value, index| {
        std.debug.print("[{d}]={d} ", .{ index, value });
    }
    std.debug.print("\n", .{});

    // Array concatenation (compile-time only)
    const a = [_]i32{ 1, 2 };
    const b = [_]i32{ 3, 4 };
    const c = a ++ b; // [1, 2, 3, 4]
    std.debug.print("Concatenated: {any}\n", .{c});

    // Array multiplication (compile-time only)
    const repeated = [_]i32{7} ** 5; // [7, 7, 7, 7, 7]
    std.debug.print("Repeated: {any}\n", .{repeated});
}

fn multidimensional_arrays() void {
    std.debug.print("\n=== Multidimensional Arrays ===\n", .{});

    // 2D array (3x4 matrix)
    const matrix = [3][4]i32{
        [_]i32{ 1, 2, 3, 4 },
        [_]i32{ 5, 6, 7, 8 },
        [_]i32{ 9, 10, 11, 12 },
    };

    // Accessing elements
    std.debug.print("matrix[1][2] = {d}\n", .{matrix[1][2]}); // 7

    // Iterating 2D array
    std.debug.print("Matrix:\n", .{});
    for (matrix) |row| {
        for (row) |value| {
            std.debug.print("{d:4}", .{value});
        }
        std.debug.print("\n", .{});
    }

    // 3D array
    const cube = [2][2][2]i32{
        [_][2]i32{
            [_]i32{ 1, 2 },
            [_]i32{ 3, 4 },
        },
        [_][2]i32{
            [_]i32{ 5, 6 },
            [_]i32{ 7, 8 },
        },
    };
    std.debug.print("cube[1][0][1] = {d}\n", .{cube[1][0][1]}); // 6
}

fn sentinel_arrays() void {
    std.debug.print("\n=== Sentinel-Terminated Arrays ===\n", .{});

    // Null-terminated string (sentinel = 0)
    const c_string: [:0]const u8 = "Hello";
    std.debug.print("C string: {s}\n", .{c_string});
    std.debug.print("Length: {d}\n", .{c_string.len});

    // Custom sentinel
    const sentinel_arr: [5:0]i32 = [_:0]i32{ 1, 2, 3, 4, 5 };
    std.debug.print("Sentinel array: {any}\n", .{sentinel_arr});
    std.debug.print("Sentinel value: {d}\n", .{sentinel_arr[5]}); // Access sentinel
}
