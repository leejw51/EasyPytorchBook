// Chapter 14: Error Handling
// Zig's error union and error handling

const std = @import("std");

// Define custom error set
const FileError = error{
    FileNotFound,
    PermissionDenied,
    InvalidPath,
};

const MathError = error{
    DivisionByZero,
    Overflow,
    NegativeNumber,
};

// Function that can fail
fn divide(a: i32, b: i32) MathError!i32 {
    if (b == 0) return error.DivisionByZero;
    return @divTrunc(a, b);
}

fn sqrt(x: i32) MathError!i32 {
    if (x < 0) return error.NegativeNumber;
    var result: i32 = 0;
    while (result * result <= x) : (result += 1) {}
    return result - 1;
}

pub fn main() void {
    basic_errors();
    try_catch();
    error_propagation();
    errdefer_example();
}

fn basic_errors() void {
    std.debug.print("=== Basic Errors ===\n", .{});

    // Call function that might fail
    const result1 = divide(10, 2);
    const result2 = divide(10, 0);

    // Check if error or value
    if (result1) |value| {
        std.debug.print("10 / 2 = {d}\n", .{value});
    } else |err| {
        std.debug.print("Error: {}\n", .{err});
    }

    if (result2) |value| {
        std.debug.print("10 / 0 = {d}\n", .{value});
    } else |err| {
        std.debug.print("10 / 0 failed: {}\n", .{err});
    }
}

fn try_catch() void {
    std.debug.print("\n=== Try and Catch ===\n", .{});

    // Catch with default value
    const safe_div = divide(10, 0) catch 0;
    std.debug.print("Safe divide (default 0): {d}\n", .{safe_div});

    // Catch with error handling
    const result = divide(10, 0) catch |err| blk: {
        std.debug.print("Caught error: {}\n", .{err});
        break :blk -1;
    };
    std.debug.print("Result with catch: {d}\n", .{result});

    // Catch unreachable (assert no error)
    const must_succeed = divide(20, 4) catch unreachable;
    std.debug.print("Must succeed: {d}\n", .{must_succeed});

    // Switch on error
    const val = sqrt(-5) catch |err| switch (err) {
        error.NegativeNumber => blk: {
            std.debug.print("Cannot sqrt negative number\n", .{});
            break :blk 0;
        },
        else => 0,
    };
    std.debug.print("sqrt result: {d}\n", .{val});
}

// Function that propagates errors
fn calculate(a: i32, b: i32) MathError!i32 {
    const quotient = try divide(a, b); // Propagate error if any
    const root = try sqrt(quotient); // Propagate error if any
    return root;
}

fn error_propagation() void {
    std.debug.print("\n=== Error Propagation ===\n", .{});

    // Try valid calculation
    if (calculate(100, 4)) |result| {
        std.debug.print("calculate(100, 4) = {d}\n", .{result});
    } else |err| {
        std.debug.print("Error: {}\n", .{err});
    }

    // Try with division by zero
    if (calculate(100, 0)) |result| {
        std.debug.print("calculate(100, 0) = {d}\n", .{result});
    } else |err| {
        std.debug.print("calculate(100, 0) error: {}\n", .{err});
    }

    // Try with negative result
    if (calculate(-25, 1)) |result| {
        std.debug.print("calculate(-25, 1) = {d}\n", .{result});
    } else |err| {
        std.debug.print("calculate(-25, 1) error: {}\n", .{err});
    }
}

// Resource with error deferred cleanup
const Resource = struct {
    id: u32,
    active: bool,

    fn init(id: u32) Resource {
        std.debug.print("Resource {d} acquired\n", .{id});
        return .{ .id = id, .active = true };
    }

    fn deinit(self: *Resource) void {
        std.debug.print("Resource {d} released\n", .{self.id});
        self.active = false;
    }
};

fn process_resource(should_fail: bool) !u32 {
    var res = Resource.init(42);

    // Cleanup only on error
    errdefer res.deinit();

    if (should_fail) {
        return error.ProcessingFailed;
    }

    // Normal path - caller responsible for cleanup
    std.debug.print("Processing succeeded\n", .{});
    res.deinit(); // Manual cleanup on success
    return res.id;
}

fn errdefer_example() void {
    std.debug.print("\n=== Errdefer Example ===\n", .{});

    // Success case
    std.debug.print("--- Success case ---\n", .{});
    if (process_resource(false)) |id| {
        std.debug.print("Got resource id: {d}\n", .{id});
    } else |_| {}

    // Error case
    std.debug.print("--- Error case ---\n", .{});
    if (process_resource(true)) |id| {
        std.debug.print("Got resource id: {d}\n", .{id});
    } else |err| {
        std.debug.print("Failed: {}\n", .{err});
    }
}

// Error union with payload
const ParseError = error{
    InvalidCharacter,
    Overflow,
};

fn parseDigit(char: u8) ParseError!u8 {
    if (char >= '0' and char <= '9') {
        return char - '0';
    }
    return error.InvalidCharacter;
}

pub fn parsing_example() void {
    std.debug.print("\n=== Parsing Example ===\n", .{});

    const inputs = [_]u8{ '5', 'a', '9', '!' };

    for (inputs) |c| {
        if (parseDigit(c)) |digit| {
            std.debug.print("'{c}' -> {d}\n", .{ c, digit });
        } else |err| {
            std.debug.print("'{c}' -> error: {}\n", .{ c, err });
        }
    }
}

// Merging error sets
const CombinedError = FileError || MathError;

fn complex_operation() CombinedError!void {
    // Can return any error from either set
    return error.FileNotFound;
}
