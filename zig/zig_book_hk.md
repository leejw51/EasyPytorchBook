# 易學 Zig

*Zig 程式語言嘅簡單實用指南*

---

## 目錄

- [0. Hello World](#0-hello-world)
- [1. Zig 簡介](#1-zig-簡介)
- [2. 變量同類型](#2-變量同類型)
- [3. 控制流程](#3-控制流程)
- [4. 函數](#4-函數)
- [5. 陣列同切片](#5-陣列同切片)
- [6. 結構體](#6-結構體)
- [7. 枚舉同聯合體](#7-枚舉同聯合體)
- [8. 錯誤處理](#8-錯誤處理)
- [9. 可選值](#9-可選值)
- [10. 指針](#10-指針)
- [11. 記憶體分配](#11-記憶體分配)
- [12. 編譯時執行](#12-編譯時執行)
- [13. 測試](#13-測試)
- [14. 構建系統](#14-構建系統)

---

## 0. Hello World


由最簡單嘅 Zig 程式開始，然後睇吓記憶體分配。

## 點樣運行

將代碼保存到文件（例如 `hello.zig`），然後用以下命令運行：

```bash
zig run hello.zig
```

就係咁簡單！Zig 一個命令就可以編譯同運行你嘅程式。

## 其他構建方法

```
┌─────────────────────────────────────────┐
│  zig run file.zig      # 編譯後運行     │
│  zig build-exe file.zig # 只係編譯      │
│  ./file                 # 運行二進制文件 │
└─────────────────────────────────────────┘
```


```
運行 Zig 程式:

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  hello.zig  │ -> │  zig run    │ -> │    輸出     │
│   (源碼)    │    │   (編譯)    │    │   (結果)    │
└─────────────┘    └─────────────┘    └─────────────┘

用 defer 嘅記憶體管理:

    ┌────────────────────────────────┐
    │  const x = try allocate();     │
    │  defer free(x);  <- 已登記     │
    │                                │
    │  ... 使用 x ...                │
    │                                │
    └────────────────────────────────┘
              │
              ▼ (作用域結束)
    ┌────────────────────────────────┐
    │  free(x) <- 執行咗!            │
    └────────────────────────────────┘
```


### 範例: hello_world

最簡單嘅 Zig 程式 - 喺控制台輸出 Hello World

```zig
const std = @import("std");

pub fn main() void {
    std.debug.print("Hello, World!\n", .{});
}
```

**輸出:**
```
Hello, World!
```


### 範例: allocator_example

用 defer 自動清理同記憶體分配嘅綜合範例

```zig
const std = @import("std");

// Function that takes an allocator and creates a dynamic string
fn createGreeting(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    // Allocate memory for the greeting
    const greeting = try std.fmt.allocPrint(allocator, "Hello, {s}! Welcome to Zig.", .{name});
    return greeting;
}

// Function that creates a dynamic array
fn createNumbers(allocator: std.mem.Allocator, count: usize) ![]u32 {
    // Allocate an array of u32
    const numbers = try allocator.alloc(u32, count);

    // Fill with values
    for (numbers, 0..) |*num, i| {
        num.* = @intCast(i * 10);
    }

    return numbers;
}

pub fn main() !void {
    // Create a General Purpose Allocator (GPA)
    // GPA detects memory leaks and double-frees in debug mode
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const status = gpa.deinit();
        if (status == .leak) {
            std.debug.print("Memory leak detected!\n", .{});
        }
    }

    const allocator = gpa.allocator();

    // Example 1: Using defer for automatic cleanup
    std.debug.print("=== Example 1: String with defer ===\n", .{});
    {
        const greeting = try createGreeting(allocator, "Alice");
        defer allocator.free(greeting); // Automatically freed when scope ends

        std.debug.print("{s}\n", .{greeting});
    } // greeting is freed here

    // Example 2: Dynamic array with defer
    std.debug.print("\n=== Example 2: Array with defer ===\n", .{});
    {
        const numbers = try createNumbers(allocator, 5);
        defer allocator.free(numbers); // Automatically freed when scope ends

        std.debug.print("Numbers: ", .{});
        for (numbers) |n| {
            std.debug.print("{d} ", .{n});
        }
        std.debug.print("\n", .{});
    } // numbers is freed here

    // Example 3: Multiple allocations with defer (LIFO order)
    std.debug.print("\n=== Example 3: Multiple defers (LIFO order) ===\n", .{});
    {
        const str1 = try createGreeting(allocator, "Bob");
        defer {
            std.debug.print("Freeing str1\n", .{});
            allocator.free(str1);
        }

        const str2 = try createGreeting(allocator, "Charlie");
        defer {
            std.debug.print("Freeing str2\n", .{});
            allocator.free(str2);
        }

        std.debug.print("{s}\n", .{str1});
        std.debug.print("{s}\n", .{str2});
    } // str2 freed first, then str1 (LIFO)

    // Example 4: Using ArrayListUnmanaged with allocator
    std.debug.print("\n=== Example 4: ArrayListUnmanaged ===\n", .{});
    {
        var list: std.ArrayListUnmanaged(i32) = .{};
        defer list.deinit(allocator); // Pass allocator to deinit

        try list.append(allocator, 100);
        try list.append(allocator, 200);
        try list.append(allocator, 300);

        std.debug.print("ArrayList items: ", .{});
        for (list.items) |item| {
            std.debug.print("{d} ", .{item});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("\n=== All memory properly freed! ===\n", .{});
}
```

**輸出:**
```
=== Example 1: String with defer ===
Hello, Alice! Welcome to Zig.

=== Example 2: Array with defer ===
Numbers: 0 10 20 30 40 

=== Example 3: Multiple defers (LIFO order) ===
Hello, Bob! Welcome to Zig.
Hello, Charlie! Welcome to Zig.
Freeing str2
Freeing str1

=== Example 4: ArrayListUnmanaged ===
ArrayList items: 100 200 300 

=== All memory properly freed! ===
```


---

## 1. Zig 簡介


Zig 係一種為以下目的設計嘅現代系統程式語言：
- **性能**: 冇隱藏嘅控制流或分配
- **安全性**: 可選嘅安全檢查，冇未定義行為
- **簡潔性**: 冇隱藏嘅魔法，代碼易讀
- **互操作性**: 直接兼容 C ABI

## 開始使用

從 https://ziglang.org/download/ 安裝 Zig

```
┌─────────────────────────────────────────┐
│           Zig 工作流程                   │
├─────────────────────────────────────────┤
│  1. 寫代碼    →  main.zig               │
│  2. 編譯      →  zig build-exe          │
│  3. 運行      →  ./main                 │
│                                         │
│  或者簡單啲:  →  zig run main.zig       │
└─────────────────────────────────────────┘
```


```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│     源碼     │ -> │    編譯器    │ -> │   二進制     │
│  main.zig    │    │     zig      │    │   ./main     │
└──────────────┘    └──────────────┘    └──────────────┘
```


### 範例: hello_world

經典嘅 Hello World 程式

```zig
const std = @import("std");

// This is your first Zig program!
// std.debug.print outputs to stderr
pub fn main() void {
    std.debug.print("Hello, Zig!\n", .{});
}
```

**輸出:**
```
Hello, Zig!
```


---

## 2. 變量同類型


Zig 有一個帶顯式類型嘅簡單類型系統。

## 變量聲明

```
┌─────────────────────────────────────────┐
│  const x = 5;    // 不可變 (const)      │
│  var y = 10;     // 可變 (var)          │
│  var z: i32 = 0; // 顯式類型            │
└─────────────────────────────────────────┘
```

## 基本類型

| 類型     | 描述              | 大小    |
|----------|-------------------|---------|
| i8-i128  | 有符號整數        | 1-16 B  |
| u8-u128  | 無符號整數        | 1-16 B  |
| f32,f64  | 浮點數            | 4,8 B   |
| bool     | 布爾值            | 1 B     |
| void     | 冇值              | 0 B     |


```
記憶體佈局 (小端序):

u8:   [xxxxxxxx]           (1 字節)
u16:  [xxxxxxxx][xxxxxxxx] (2 字節)
u32:  [xxxx][xxxx][xxxx][xxxx] (4 字節)

i8 範圍:  -128 到 127
u8 範圍:  0 到 255
```


### 範例: variables

變量聲明同基本類型

```zig
const std = @import("std");

pub fn main() void {
    // Immutable constant
    const message = "Hello";

    // Mutable variable
    var count: i32 = 0;
    count += 1;

    // Type inference
    const pi = 3.14159;

    // Explicit type
    const byte: u8 = 255;

    std.debug.print("message: {s}\n", .{message});
    std.debug.print("count: {}\n", .{count});
    std.debug.print("pi: {d:.2}\n", .{pi});
    std.debug.print("byte: {}\n", .{byte});
}
```

**輸出:**
```
message: Hello
count: 1
pi: 3.14
byte: 255
```


### 範例: type_coercion

類型轉換同強制轉換

```zig
const std = @import("std");

pub fn main() void {
    // Integer types
    const small: u8 = 10;
    const big: u32 = small; // Safe widening

    // Explicit cast for narrowing
    const large: u32 = 1000;
    const truncated: u8 = @truncate(large);

    std.debug.print("small: {}, big: {}\n", .{ small, big });
    std.debug.print("large: {}, truncated: {}\n", .{ large, truncated });
}
```

**輸出:**
```
small: 10, big: 10
large: 1000, truncated: 232
```


---

## 3. 控制流程


Zig 提供標準控制流程，仲有一啲獨特功能。

## If/Else

```
┌─────────────────────────────────────────┐
│  if (條件) {                            │
│      // 真分支                          │
│  } else {                               │
│      // 假分支                          │
│  }                                      │
└─────────────────────────────────────────┘
```

## While 循環

```
┌─────────────────────────────────────────┐
│  while (條件) : (繼續表達式) {           │
│      // 循環體                          │
│  }                                      │
└─────────────────────────────────────────┘
```


```
控制流程圖:

    ┌───────┐
    │ 開始  │
    └───┬───┘
        │
    ┌───▼───┐     係     ┌───────┐
    │ 條件? ├────────────► 區塊  │
    └───┬───┘            └───────┘
        │ 唔係
    ┌───▼───┐
    │  結束  │
    └───────┘
```


### 範例: if_else

If/else 語句同表達式

```zig
const std = @import("std");

pub fn main() void {
    const x: i32 = 42;

    // Basic if/else
    if (x > 0) {
        std.debug.print("x is positive\n", .{});
    } else if (x < 0) {
        std.debug.print("x is negative\n", .{});
    } else {
        std.debug.print("x is zero\n", .{});
    }

    // If as expression
    const abs_x = if (x < 0) -x else x;
    std.debug.print("abs(x) = {}\n", .{abs_x});
}
```

**輸出:**
```
x is positive
abs(x) = 42
```


### 範例: loops

While 同 for 循環

```zig
const std = @import("std");

pub fn main() void {
    // While loop
    var i: u32 = 0;
    while (i < 3) : (i += 1) {
        std.debug.print("while i = {}\n", .{i});
    }

    // For loop over range
    for (0..3) |j| {
        std.debug.print("for j = {}\n", .{j});
    }

    // For loop over array
    const arr = [_]i32{ 10, 20, 30 };
    for (arr) |val| {
        std.debug.print("val = {}\n", .{val});
    }
}
```

**輸出:**
```
while i = 0
while i = 1
while i = 2
for j = 0
for j = 1
for j = 2
val = 10
val = 20
val = 30
```


### 範例: switch

Switch 表達式

```zig
const std = @import("std");

pub fn main() void {
    const day: u8 = 3;

    // Switch statement
    const name = switch (day) {
        1 => "Monday",
        2 => "Tuesday",
        3 => "Wednesday",
        4 => "Thursday",
        5 => "Friday",
        6, 7 => "Weekend",
        else => "Invalid",
    };

    std.debug.print("Day {}: {s}\n", .{ day, name });
}
```

**輸出:**
```
Day 3: Wednesday
```


---

## 4. 函數


函數喺 Zig 入面係一等公民。

## 函數語法

```
┌─────────────────────────────────────────┐
│  fn 名稱(參數: 類型) 返回類型 {          │
│      return 值;                         │
│  }                                      │
└─────────────────────────────────────────┘
```

## 特殊功能
- 函數可以作為值傳遞
- 編譯時函數求值
- 用 `inline` 嘅內聯函數


```
函數調用棧:

┌─────────────────────┐
│   main()            │  <- 棧幀
├─────────────────────┤
│   add(3, 4)         │  <- 新幀
│   a = 3, b = 4      │
│   return 7          │
├─────────────────────┤
│   (返回到 main)      │
└─────────────────────┘
```


### 範例: basic_functions

基本函數定義

```zig
const std = @import("std");

// Simple function
fn add(a: i32, b: i32) i32 {
    return a + b;
}

// Function with multiple return (tuple)
fn divmod(a: i32, b: i32) struct { quot: i32, rem: i32 } {
    return .{ .quot = @divTrunc(a, b), .rem = @mod(a, b) };
}

pub fn main() void {
    const sum = add(3, 4);
    std.debug.print("3 + 4 = {}\n", .{sum});

    const result = divmod(17, 5);
    std.debug.print("17 / 5 = {} remainder {}\n", .{ result.quot, result.rem });
}
```

**輸出:**
```
3 + 4 = 7
17 / 5 = 3 remainder 2
```


### 範例: function_pointers

函數指針同高階函數

```zig
const std = @import("std");

fn double(x: i32) i32 {
    return x * 2;
}

fn triple(x: i32) i32 {
    return x * 3;
}

// Function that takes a function pointer
fn apply(value: i32, func: *const fn (i32) i32) i32 {
    return func(value);
}

pub fn main() void {
    const x: i32 = 5;

    std.debug.print("double({}) = {}\n", .{ x, apply(x, &double) });
    std.debug.print("triple({}) = {}\n", .{ x, apply(x, &triple) });
}
```

**輸出:**
```
double(5) = 10
triple(5) = 15
```


---

## 5. 陣列同切片


陣列有固定大小；切片係陣列嘅視圖。

## 陣列 vs 切片

```
┌─────────────────────────────────────────┐
│  陣列: [N]T   - 編譯時固定大小          │
│  切片: []T    - 運行時陣列視圖          │
└─────────────────────────────────────────┘
```


```
記憶體中嘅陣列:
┌───┬───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │  arr[5]i32
└───┴───┴───┴───┴───┘
  0   1   2   3   4

切片 (視圖):
        ┌───────────┐
        │ ptr + len │
        └─────┬─────┘
              │
┌───┬───┬───┬─▼─┬───┐
│ 1 │ 2 │ 3 │ 4 │ 5 │
└───┴───┴───┴───┴───┘
        └─────┘
        slice [1..4]
```


### 範例: arrays

陣列基礎

```zig
const std = @import("std");

pub fn main() void {
    // Array with explicit size
    const arr1: [5]i32 = [_]i32{ 1, 2, 3, 4, 5 };

    // Array with inferred size
    const arr2 = [_]i32{ 10, 20, 30 };

    // Access elements
    std.debug.print("arr1[0] = {}\n", .{arr1[0]});
    std.debug.print("arr2.len = {}\n", .{arr2.len});

    // Iterate
    var sum: i32 = 0;
    for (arr1) |x| {
        sum += x;
    }
    std.debug.print("sum of arr1 = {}\n", .{sum});
}
```

**輸出:**
```
arr1[0] = 1
arr2.len = 3
sum of arr1 = 15
```


### 範例: slices

切片 - 陣列嘅視圖

```zig
const std = @import("std");

pub fn main() void {
    const arr = [_]i32{ 1, 2, 3, 4, 5 };

    // Create slice from array
    const slice: []const i32 = arr[1..4]; // [2, 3, 4]

    std.debug.print("slice.len = {}\n", .{slice.len});

    for (slice, 0..) |val, i| {
        std.debug.print("slice[{}] = {}\n", .{ i, val });
    }
}
```

**輸出:**
```
slice.len = 3
slice[0] = 2
slice[1] = 3
slice[2] = 4
```


### 範例: string_literals

字符串作為字節切片

```zig
const std = @import("std");

pub fn main() void {
    // String literal is []const u8
    const hello = "Hello, World!";

    std.debug.print("String: {s}\n", .{hello});
    std.debug.print("Length: {}\n", .{hello.len});
    std.debug.print("First char: {c}\n", .{hello[0]});

    // Iterate over characters
    for (hello[0..5]) |c| {
        std.debug.print("{c} ", .{c});
    }
    std.debug.print("\n", .{});
}
```

**輸出:**
```
String: Hello, World!
Length: 13
First char: H
H e l l o
```


---

## 6. 結構體


結構體將相關數據組合埋一齊。

## 結構體定義

```
┌─────────────────────────────────────────┐
│  const Point = struct {                 │
│      x: i32,                            │
│      y: i32,                            │
│                                         │
│      fn init(x: i32, y: i32) Point {    │
│          return .{ .x = x, .y = y };    │
│      }                                  │
│  };                                     │
└─────────────────────────────────────────┘
```


```
結構體記憶體佈局:

Point { x: i32, y: i32 }

┌─────────────┬─────────────┐
│    x: i32   │    y: i32   │
│   4 字節    │   4 字節    │
└─────────────┴─────────────┘
     總共: 8 字節
```


### 範例: basic_struct

帶方法嘅結構體定義

```zig
const std = @import("std");

// Define a struct
const Point = struct {
    x: i32,
    y: i32,

    // Method
    fn distance_from_origin(self: Point) f32 {
        const fx: f32 = @floatFromInt(self.x);
        const fy: f32 = @floatFromInt(self.y);
        return @sqrt(fx * fx + fy * fy);
    }
};

pub fn main() void {
    // Create instance
    const p = Point{ .x = 3, .y = 4 };

    std.debug.print("Point: ({}, {})\n", .{ p.x, p.y });
    std.debug.print("Distance: {d:.2}\n", .{p.distance_from_origin()});
}
```

**輸出:**
```
Point: (3, 4)
Distance: 5.00
```


### 範例: struct_defaults

帶默認字段值嘅結構體

```zig
const std = @import("std");

const Config = struct {
    // Fields with default values
    width: u32 = 800,
    height: u32 = 600,
    title: []const u8 = "Untitled",
};

pub fn main() void {
    // Use all defaults
    const c1 = Config{};

    // Override some defaults
    const c2 = Config{ .width = 1920, .height = 1080 };

    std.debug.print("c1: {}x{} '{s}'\n", .{ c1.width, c1.height, c1.title });
    std.debug.print("c2: {}x{} '{s}'\n", .{ c2.width, c2.height, c2.title });
}
```

**輸出:**
```
c1: 800x600 'Untitled'
c2: 1920x1080 'Untitled'
```


---

## 7. 枚舉同聯合體


枚舉定義一組命名值。標記聯合體將枚舉同數據結合。

## 枚舉

```
┌─────────────────────────────────────────┐
│  const Color = enum { red, green, blue };│
└─────────────────────────────────────────┘
```

## 標記聯合體

```
┌─────────────────────────────────────────┐
│  const Value = union(enum) {            │
│      int: i32,                          │
│      float: f64,                        │
│      none,                              │
│  };                                     │
└─────────────────────────────────────────┘
```


```
標記聯合體記憶體:

Value union(enum):
┌─────────┬─────────────────┐
│   標記  │      數據       │
│ (enum)  │ (最大嘅字段)    │
└─────────┴─────────────────┘

標記值: .int=0, .float=1, .boolean=2, .none=3
```


### 範例: enums

帶方法嘅枚舉

```zig
const std = @import("std");

const Direction = enum {
    north,
    south,
    east,
    west,

    // Enum method
    fn opposite(self: Direction) Direction {
        return switch (self) {
            .north => .south,
            .south => .north,
            .east => .west,
            .west => .east,
        };
    }
};

pub fn main() void {
    const dir = Direction.north;
    const opp = dir.opposite();

    std.debug.print("Direction: {}\n", .{dir});
    std.debug.print("Opposite: {}\n", .{opp});
}
```

**輸出:**
```
Direction: .north
Opposite: .south
```


### 範例: tagged_union

類型安全變體嘅標記聯合體

```zig
const std = @import("std");

const Value = union(enum) {
    int: i32,
    float: f64,
    boolean: bool,
    none: void,
};

fn printValue(val: Value) void {
    switch (val) {
        .int => |i| std.debug.print("int: {}\n", .{i}),
        .float => |f| std.debug.print("float: {d:.2}\n", .{f}),
        .boolean => |b| std.debug.print("bool: {}\n", .{b}),
        .none => std.debug.print("none\n", .{}),
    }
}

pub fn main() void {
    const v1 = Value{ .int = 42 };
    const v2 = Value{ .float = 3.14 };
    const v3 = Value{ .boolean = true };
    const v4 = Value{ .none = {} };

    printValue(v1);
    printValue(v2);
    printValue(v3);
    printValue(v4);
}
```

**輸出:**
```
int: 42
float: 3.14
bool: true
none
```


---

## 8. 錯誤處理


Zig 使用錯誤聯合體進行顯式錯誤處理。

## 錯誤聯合體類型

```
┌─────────────────────────────────────────┐
│  fn divide(a: i32, b: i32) !i32 {       │
│      if (b == 0) return error.DivByZero;│
│      return @divTrunc(a, b);            │
│  }                                      │
└─────────────────────────────────────────┘
```

## 錯誤處理選項
- `try`: 將錯誤向上傳播
- `catch`: 處理錯誤
- `orelse`: 錯誤時嘅默認值


```
錯誤聯合體類型:

    anyerror!T 或 ErrorSet!T
         │
    ┌────┴────┐
    │         │
  錯誤       值
    │         │
┌───▼───┐ ┌───▼───┐
│ catch │ │ 成功  │
└───────┘ └───────┘
```


### 範例: error_basics

用 catch 嘅基本錯誤處理

```zig
const std = @import("std");

const MathError = error{
    DivisionByZero,
    Overflow,
};

fn divide(a: i32, b: i32) MathError!i32 {
    if (b == 0) return MathError.DivisionByZero;
    return @divTrunc(a, b);
}

pub fn main() void {
    // Using catch to handle error
    const result1 = divide(10, 2) catch |err| {
        std.debug.print("Error: {}\n", .{err});
        return;
    };
    std.debug.print("10 / 2 = {}\n", .{result1});

    // Using catch with default
    const result2 = divide(10, 0) catch 0;
    std.debug.print("10 / 0 = {} (default)\n", .{result2});
}
```

**輸出:**
```
10 / 2 = 5
10 / 0 = 0 (default)
```


### 範例: try_keyword

用 try 傳播錯誤

```zig
const std = @import("std");

const ParseError = error{InvalidChar};

fn parseDigit(c: u8) ParseError!u8 {
    if (c >= '0' and c <= '9') {
        return c - '0';
    }
    return ParseError.InvalidChar;
}

fn parseNumber(s: []const u8) ParseError!u32 {
    var result: u32 = 0;
    for (s) |c| {
        // try propagates error if parseDigit fails
        const digit = try parseDigit(c);
        result = result * 10 + digit;
    }
    return result;
}

pub fn main() void {
    if (parseNumber("123")) |num| {
        std.debug.print("Parsed: {}\n", .{num});
    } else |err| {
        std.debug.print("Parse error: {}\n", .{err});
    }

    if (parseNumber("12x")) |num| {
        std.debug.print("Parsed: {}\n", .{num});
    } else |err| {
        std.debug.print("Parse error: {}\n", .{err});
    }
}
```

**輸出:**
```
Parsed: 123
Parse error: error.InvalidChar
```


---

## 9. 可選值


可選值表示可能存在或唔存在嘅值。

## 可選類型

```
┌─────────────────────────────────────────┐
│  var x: ?i32 = null;   // 冇值          │
│  x = 42;               // 有值          │
│                                         │
│  if (x) |val| {        // 解包          │
│      // 使用 val                        │
│  }                                      │
└─────────────────────────────────────────┘
```


```
可選類型 ?T:

┌─────────────┐
│   ?i32      │
└──────┬──────┘
       │
   ┌───┴───┐
   │       │
 null     值
   │       │
┌──▼──┐ ┌──▼──┐
│ 空  │ │ 42  │
└─────┘ └─────┘
```


### 範例: optionals

可選類型同解包

```zig
const std = @import("std");

fn find_first_even(arr: []const i32) ?i32 {
    for (arr) |x| {
        if (@mod(x, 2) == 0) return x;
    }
    return null; // Not found
}

pub fn main() void {
    const arr1 = [_]i32{ 1, 3, 4, 7, 9 };
    const arr2 = [_]i32{ 1, 3, 5, 7, 9 };

    // Using if to unwrap
    if (find_first_even(&arr1)) |val| {
        std.debug.print("First even: {}\n", .{val});
    } else {
        std.debug.print("No even number found\n", .{});
    }

    // Using orelse for default
    const result = find_first_even(&arr2) orelse -1;
    std.debug.print("Result with default: {}\n", .{result});
}
```

**輸出:**
```
First even: 4
Result with default: -1
```


### 範例: optional_pointers

用於鏈接結構嘅可選指針

```zig
const std = @import("std");

const Node = struct {
    value: i32,
    next: ?*Node, // Optional pointer
};

pub fn main() void {
    var node3 = Node{ .value = 30, .next = null };
    var node2 = Node{ .value = 20, .next = &node3 };
    var node1 = Node{ .value = 10, .next = &node2 };

    // Traverse linked list
    var current: ?*Node = &node1;
    while (current) |node| {
        std.debug.print("Value: {}\n", .{node.value});
        current = node.next;
    }
}
```

**輸出:**
```
Value: 10
Value: 20
Value: 30
```


---

## 10. 指針


Zig 有唔同嘅指針類型用於唔同嘅用例。

## 指針類型

```
┌─────────────────────────────────────────┐
│  *T       - 單項指針                    │
│  [*]T     - 多項指針                    │
│  *const T - 指向 const 嘅指針           │
│  ?*T      - 可選指針                    │
└─────────────────────────────────────────┘
```


```
指向變量嘅指針:

   ptr          x
┌───────┐   ┌───────┐
│ 地址 ─┼──►│  10   │
└───────┘   └───────┘

*ptr 解引用獲取值
&x 獲取 x 嘅地址
```


### 範例: pointers_basic

基本指針操作

```zig
const std = @import("std");

pub fn main() void {
    var x: i32 = 10;

    // Get pointer
    const ptr: *i32 = &x;

    // Dereference
    std.debug.print("x = {}\n", .{x});
    std.debug.print("*ptr = {}\n", .{ptr.*});

    // Modify through pointer
    ptr.* = 20;
    std.debug.print("After modification: x = {}\n", .{x});
}
```

**輸出:**
```
x = 10
*ptr = 10
After modification: x = 20
```


### 範例: pointer_arithmetic

指針算術同切片

```zig
const std = @import("std");

pub fn main() void {
    const arr = [_]i32{ 10, 20, 30, 40, 50 };

    // Many-item pointer
    const ptr: [*]const i32 = &arr;

    // Access by index
    std.debug.print("ptr[0] = {}\n", .{ptr[0]});
    std.debug.print("ptr[2] = {}\n", .{ptr[2]});

    // Slice from pointer
    const slice = ptr[1..4];
    for (slice) |val| {
        std.debug.print("slice val: {}\n", .{val});
    }
}
```

**輸出:**
```
ptr[0] = 10
ptr[2] = 30
slice val: 20
slice val: 30
slice val: 40
```


---

## 11. 記憶體分配


Zig 俾你對記憶體分配有顯式控制。

## 分配器接口

```
┌─────────────────────────────────────────┐
│  const allocator = std.heap.page_allocator;│
│  const ptr = try allocator.create(T);   │
│  defer allocator.destroy(ptr);          │
└─────────────────────────────────────────┘
```

## 常見分配器
- `page_allocator`: 操作系統頁面分配
- `GeneralPurposeAllocator`: 調試友好
- `ArenaAllocator`: 批量釋放


```
記憶體管理:

棧 (自動):
┌─────────────────┐
│  局部變量       │ <- 快速，自動清理
└─────────────────┘

堆 (手動):
┌─────────────────┐
│ allocator.alloc │ <- 顯式分配
│ allocator.free  │ <- 顯式釋放
└─────────────────┘

用 defer 清理!
```


### 範例: allocation

動態記憶體分配

```zig
const std = @import("std");

pub fn main() !void {
    // Use general purpose allocator
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Allocate array
    const arr = try allocator.alloc(i32, 5);
    defer allocator.free(arr);

    // Initialize
    for (arr, 0..) |*item, i| {
        item.* = @intCast(i * 10);
    }

    // Print
    for (arr) |val| {
        std.debug.print("{} ", .{val});
    }
    std.debug.print("\n", .{});
}
```

**輸出:**
```
0 10 20 30 40
```


### 範例: arraylist

用 ArrayList 做動態陣列

```zig
const std = @import("std");

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // ArrayList - dynamic array
    var list: std.ArrayListUnmanaged(i32) = .empty;
    defer list.deinit(allocator);

    // Add elements
    try list.append(allocator, 10);
    try list.append(allocator, 20);
    try list.append(allocator, 30);

    std.debug.print("List: ", .{});
    for (list.items) |val| {
        std.debug.print("{} ", .{val});
    }
    std.debug.print("\n", .{});
    std.debug.print("Length: {}\n", .{list.items.len});
}
```

**輸出:**
```
List: 10 20 30 
Length: 3
```


---

## 12. 編譯時執行


Zig 可以用 `comptime` 喺編譯時執行代碼。

## Comptime 功能

```
┌─────────────────────────────────────────┐
│  comptime {                             │
│      // 喺編譯時運行                    │
│  }                                      │
│                                         │
│  fn generic(comptime T: type) type {    │
│      // 類型級別計算                    │
│  }                                      │
└─────────────────────────────────────────┘
```


```
編譯時 vs 運行時:

┌──────────────────┐     ┌──────────────────┐
│     編譯時       │     │      運行時      │
├──────────────────┤     ├──────────────────┤
│ comptime 區塊    │     │ 普通代碼         │
│ 類型計算         │     │ 用戶輸入         │
│ 泛型參數         │     │ 動態數據         │
│ 常量折疊         │     │ 堆分配           │
└──────────────────┘     └──────────────────┘
        │                        │
        └────────┬───────────────┘
                 │
            最終二進制
```


### 範例: comptime_basic

編譯時計算

```zig
const std = @import("std");

// Compile-time function
fn factorial(n: u64) u64 {
    if (n == 0) return 1;
    return n * factorial(n - 1);
}

// Const values with comptime-known args are computed at compile time
const factorial_10 = factorial(10);

pub fn main() void {
    // This value is computed at compile time!
    std.debug.print("10! = {}\n", .{factorial_10});

    // Comptime type introspection
    const T = i32;
    std.debug.print("Size of i32: {} bytes\n", .{@sizeOf(T)});

    // Inline comptime block
    const squares = comptime blk: {
        var arr: [5]i32 = undefined;
        for (0..5) |i| {
            arr[i] = @as(i32, @intCast(i)) * @as(i32, @intCast(i));
        }
        break :blk arr;
    };
    std.debug.print("Squares: ", .{});
    for (squares) |s| {
        std.debug.print("{} ", .{s});
    }
    std.debug.print("\n", .{});
}
```

**輸出:**
```
10! = 3628800
Size of i32: 4 bytes
Squares: 0 1 4 9 16
```


### 範例: generic_function

用 comptime 嘅泛型函數同類型

```zig
const std = @import("std");

// Generic max function using comptime
fn max(comptime T: type, a: T, b: T) T {
    return if (a > b) a else b;
}

// Generic struct
fn Pair(comptime T: type) type {
    return struct {
        first: T,
        second: T,

        fn swap(self: *@This()) void {
            const tmp = self.first;
            self.first = self.second;
            self.second = tmp;
        }
    };
}

pub fn main() void {
    // Works with any comparable type
    std.debug.print("max(3, 7) = {}\n", .{max(i32, 3, 7)});
    std.debug.print("max(3.5, 2.1) = {d:.1}\n", .{max(f64, 3.5, 2.1)});

    // Generic struct usage
    var pair = Pair(i32){ .first = 10, .second = 20 };
    std.debug.print("Before swap: {}, {}\n", .{ pair.first, pair.second });
    pair.swap();
    std.debug.print("After swap: {}, {}\n", .{ pair.first, pair.second });
}
```

**輸出:**
```
max(3, 7) = 7
max(3.5, 2.1) = 3.5
Before swap: 10, 20
After swap: 20, 10
```


---

## 13. 測試


Zig 有內置測試支持。

## 測試語法

```
┌─────────────────────────────────────────┐
│  test "描述" {                          │
│      try std.testing.expect(true);      │
│      try std.testing.expectEqual(1, 1); │
│  }                                      │
└─────────────────────────────────────────┘
```

運行測試: `zig test file.zig`


```
測試工作流程:

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  寫測試     │ -> │  zig test   │ -> │    結果     │
└─────────────┘    └─────────────┘    └─────────────┘
                          │
                   ┌──────┴──────┐
                   │             │
                 通過          失敗
                   │             │
                  [OK]      [錯誤信息]
```


### 範例: testing

編寫同運行測試

```zig
const std = @import("std");

fn add(a: i32, b: i32) i32 {
    return a + b;
}

fn divide(a: i32, b: i32) !i32 {
    if (b == 0) return error.DivisionByZero;
    return @divTrunc(a, b);
}

test "add function" {
    try std.testing.expectEqual(add(2, 3), 5);
    try std.testing.expectEqual(add(-1, 1), 0);
}

test "divide function" {
    const result = try divide(10, 2);
    try std.testing.expectEqual(result, 5);
}

test "divide by zero returns error" {
    const result = divide(10, 0);
    try std.testing.expectError(error.DivisionByZero, result);
}

pub fn main() void {
    std.debug.print("Run 'zig test' to execute tests\n", .{});
    std.debug.print("add(2,3) = {}\n", .{add(2, 3)});
}
```

**輸出:**
```
Run 'zig test' to execute tests
add(2,3) = 5
```


---

## 14. 構建系統


Zig 有一個用 `build.zig` 嘅強大內置構建系統。

## 基本 build.zig

```
┌─────────────────────────────────────────┐
│  zig init     - 創建新項目              │
│  zig build    - 構建項目                │
│  zig build run - 構建後運行             │
└─────────────────────────────────────────┘
```


```
項目結構:

my_project/
├── build.zig       <- 構建配置
├── build.zig.zon   <- 依賴
└── src/
    ├── main.zig    <- 入口點
    └── lib.zig     <- 庫代碼

構建流程:
┌────────────┐   ┌────────────┐   ┌────────────┐
│ build.zig  │ → │ zig build  │ → │ zig-out/   │
└────────────┘   └────────────┘   └────────────┘
```


### 範例: build_example

項目結構同構建命令

```zig
const std = @import("std");

// Example of a library function
pub fn greet(name: []const u8) void {
    std.debug.print("Hello, {s}!\n", .{name});
}

// Example of configuration
pub const Config = struct {
    debug: bool = false,
    optimization: enum { none, speed, size } = .none,
};

pub fn main() void {
    std.debug.print("Build System Demo\n", .{});
    std.debug.print("================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Commands:\n", .{});
    std.debug.print("  zig init        - Create new project\n", .{});
    std.debug.print("  zig build       - Build project\n", .{});
    std.debug.print("  zig build run   - Build and run\n", .{});
    std.debug.print("  zig build test  - Run tests\n", .{});
    std.debug.print("\n", .{});
    greet("World");
}
```

**輸出:**
```
Build System Demo
================

Commands:
  zig init        - Create new project
  zig build       - Build project
  zig build run   - Build and run
  zig build test  - Run tests

Hello, World!
```


---

## 快速參考


### 常用命令
```
zig run file.zig     # 編譯後運行
zig build-exe file.zig  # 編譯為執行文件
zig test file.zig    # 運行測試
zig fmt file.zig     # 格式化代碼
zig init             # 初始化新項目
zig build            # 構建項目
```

### 有用嘅內置函數
| 內置函數 | 描述 |
|----------|-------------|
| @import | 導入模塊 |
| @intCast | 整數類型轉換 |
| @floatFromInt | 整數轉浮點數 |
| @truncate | 截斷為更小類型 |
| @sizeOf | 獲取類型大小 |
| @TypeOf | 獲取表達式類型 |

### 格式說明符
| 說明符 | 描述 |
|-----------|-------------|
| {} | 默認 |
| {s} | 字符串 |
| {c} | 字符 |
| {d} | 十進制 |
| {x} | 十六進制 |
| {b} | 二進制 |
