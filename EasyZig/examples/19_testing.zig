// Chapter 19: Testing
// Built-in test framework in Zig

const std = @import("std");
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;
const expectError = std.testing.expectError;

// Function to test
fn add(a: i32, b: i32) i32 {
    return a + b;
}

fn divide(a: i32, b: i32) !i32 {
    if (b == 0) return error.DivisionByZero;
    return @divTrunc(a, b);
}

fn factorial(n: u32) u32 {
    if (n == 0) return 1;
    return n * factorial(n - 1);
}

// Basic test
test "addition" {
    const result = add(2, 3);
    try expectEqual(@as(i32, 5), result);
}

test "addition with negatives" {
    try expectEqual(@as(i32, -1), add(2, -3));
    try expectEqual(@as(i32, 0), add(5, -5));
}

// Testing with expect
test "expect examples" {
    try expect(add(1, 1) == 2);
    try expect(10 > 5);
    try expect(true);
}

// Testing errors
test "division by zero" {
    try expectError(error.DivisionByZero, divide(10, 0));
}

test "successful division" {
    const result = try divide(10, 2);
    try expectEqual(@as(i32, 5), result);
}

// Testing with optional
fn findFirst(haystack: []const u8, needle: u8) ?usize {
    for (haystack, 0..) |byte, i| {
        if (byte == needle) return i;
    }
    return null;
}

test "find character" {
    const text = "hello";
    try expectEqual(@as(?usize, 0), findFirst(text, 'h'));
    try expectEqual(@as(?usize, 2), findFirst(text, 'l'));
    try expectEqual(@as(?usize, null), findFirst(text, 'z'));
}

// Testing factorial
test "factorial" {
    try expectEqual(@as(u32, 1), factorial(0));
    try expectEqual(@as(u32, 1), factorial(1));
    try expectEqual(@as(u32, 120), factorial(5));
    try expectEqual(@as(u32, 3628800), factorial(10));
}

// Test struct
const Counter = struct {
    value: i32 = 0,

    pub fn increment(self: *Counter) void {
        self.value += 1;
    }

    pub fn decrement(self: *Counter) void {
        self.value -= 1;
    }

    pub fn reset(self: *Counter) void {
        self.value = 0;
    }
};

test "Counter operations" {
    var counter = Counter{};

    try expectEqual(@as(i32, 0), counter.value);

    counter.increment();
    counter.increment();
    try expectEqual(@as(i32, 2), counter.value);

    counter.decrement();
    try expectEqual(@as(i32, 1), counter.value);

    counter.reset();
    try expectEqual(@as(i32, 0), counter.value);
}

// Test with allocator
test "ArrayList operations" {
    const allocator = std.testing.allocator;

    var list = std.ArrayList(i32).init(allocator);
    defer list.deinit();

    try list.append(1);
    try list.append(2);
    try list.append(3);

    try expectEqual(@as(usize, 3), list.items.len);
    try expectEqual(@as(i32, 1), list.items[0]);
    try expectEqual(@as(i32, 3), list.items[2]);
}

// Skipping tests
test "skip this test" {
    // Skip test with message
    if (true) return error.SkipZigTest;
}

// Memory leak detection (uses testing allocator)
test "no memory leaks" {
    const allocator = std.testing.allocator;

    const ptr = try allocator.create(i32);
    defer allocator.destroy(ptr);

    ptr.* = 42;
    try expectEqual(@as(i32, 42), ptr.*);
}

// Run tests: zig test 19_testing.zig
pub fn main() void {
    std.debug.print("Run tests with: zig test 19_testing.zig\n", .{});
}
