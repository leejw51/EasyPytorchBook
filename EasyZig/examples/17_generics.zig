// Chapter 17: Generics
// Generic programming with comptime

const std = @import("std");

// Generic function
fn max(comptime T: type, a: T, b: T) T {
    return if (a > b) a else b;
}

fn min(comptime T: type, a: T, b: T) T {
    return if (a < b) a else b;
}

fn swap(comptime T: type, a: *T, b: *T) void {
    const temp = a.*;
    a.* = b.*;
    b.* = temp;
}

pub fn main() void {
    generic_functions();
    generic_structs();
    generic_algorithms();
}

fn generic_functions() void {
    std.debug.print("=== Generic Functions ===\n", .{});

    // Use with different types
    std.debug.print("max(i32, 5, 10) = {d}\n", .{max(i32, 5, 10)});
    std.debug.print("max(f64, 3.14, 2.71) = {d}\n", .{max(f64, 3.14, 2.71)});
    std.debug.print("min(i32, 5, 10) = {d}\n", .{min(i32, 5, 10)});

    // Swap values
    var x: i32 = 100;
    var y: i32 = 200;
    std.debug.print("Before swap: x={d}, y={d}\n", .{ x, y });
    swap(i32, &x, &y);
    std.debug.print("After swap: x={d}, y={d}\n", .{ x, y });
}

// Generic Stack struct
fn Stack(comptime T: type) type {
    return struct {
        const Self = @This();
        const CAPACITY = 100;

        items: [CAPACITY]T = undefined,
        count: usize = 0,

        pub fn push(self: *Self, item: T) !void {
            if (self.count >= CAPACITY) return error.StackOverflow;
            self.items[self.count] = item;
            self.count += 1;
        }

        pub fn pop(self: *Self) ?T {
            if (self.count == 0) return null;
            self.count -= 1;
            return self.items[self.count];
        }

        pub fn peek(self: *const Self) ?T {
            if (self.count == 0) return null;
            return self.items[self.count - 1];
        }

        pub fn len(self: *const Self) usize {
            return self.count;
        }
    };
}

// Generic Pair struct
fn Pair(comptime T1: type, comptime T2: type) type {
    return struct {
        first: T1,
        second: T2,
    };
}

fn generic_structs() void {
    std.debug.print("\n=== Generic Structs ===\n", .{});

    // Integer stack
    var int_stack = Stack(i32){};

    int_stack.push(10) catch {};
    int_stack.push(20) catch {};
    int_stack.push(30) catch {};

    std.debug.print("Stack size: {d}\n", .{int_stack.len()});
    std.debug.print("Top: {?d}\n", .{int_stack.peek()});

    while (int_stack.pop()) |value| {
        std.debug.print("Popped: {d}\n", .{value});
    }

    // Pair with different types
    const pair = Pair(i32, []const u8){ .first = 42, .second = "answer" };
    std.debug.print("Pair: ({d}, {s})\n", .{ pair.first, pair.second });
}

// Generic search algorithm
fn linearSearch(comptime T: type, haystack: []const T, needle: T) ?usize {
    for (haystack, 0..) |item, index| {
        if (item == needle) return index;
    }
    return null;
}

// Generic reverse
fn reverse(comptime T: type, items: []T) void {
    var left: usize = 0;
    var right: usize = items.len;
    while (left < right) {
        right -= 1;
        const temp = items[left];
        items[left] = items[right];
        items[right] = temp;
        left += 1;
    }
}

// Generic count
fn count(comptime T: type, items: []const T, target: T) usize {
    var result: usize = 0;
    for (items) |item| {
        if (item == target) result += 1;
    }
    return result;
}

fn generic_algorithms() void {
    std.debug.print("\n=== Generic Algorithms ===\n", .{});

    // Search
    const nums = [_]i32{ 10, 20, 30, 40, 50 };
    if (linearSearch(i32, &nums, 30)) |idx| {
        std.debug.print("Found 30 at index {d}\n", .{idx});
    }

    // Reverse
    var arr = [_]i32{ 1, 2, 3, 4, 5 };
    std.debug.print("Before reverse: {any}\n", .{arr});
    reverse(i32, &arr);
    std.debug.print("After reverse: {any}\n", .{arr});

    // Count
    const values = [_]i32{ 1, 2, 2, 3, 2, 4, 2, 5 };
    std.debug.print("Count of 2: {d}\n", .{count(i32, &values, 2)});
}
