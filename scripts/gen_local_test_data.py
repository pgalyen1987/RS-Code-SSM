"""Generate synthetic coding examples for local SFT+GRPO integration test."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from train.grpo import build_system_prompt

SYSTEM = build_system_prompt("python")

EXAMPLES = [
    ("Write `add(a, b)` returning a+b.",
     "Simple addition.", "def add(a, b):\n    return a + b",
     "assert add(1,2)==3\nassert add(-1,1)==0\nprint('PASS')"),
    ("Write `is_palindrome(s)` returning True if s is a palindrome.",
     "Compare string to its reverse.", "def is_palindrome(s):\n    return s == s[::-1]",
     "assert is_palindrome('racecar')\nassert not is_palindrome('hello')\nprint('PASS')"),
    ("Write `factorial(n)` using recursion.",
     "Base case n<=1 returns 1.", "def factorial(n):\n    return 1 if n<=1 else n*factorial(n-1)",
     "assert factorial(5)==120\nassert factorial(0)==1\nprint('PASS')"),
    ("Write `fizzbuzz(n)` returning a list with Fizz/Buzz/FizzBuzz rules.",
     "Check divisibility by 15, then 3, then 5.",
     "def fizzbuzz(n):\n    r=[]\n    for i in range(1,n+1):\n        if i%15==0: r.append('FizzBuzz')\n        elif i%3==0: r.append('Fizz')\n        elif i%5==0: r.append('Buzz')\n        else: r.append(i)\n    return r",
     "r=fizzbuzz(15)\nassert r[2]=='Fizz'\nassert r[4]=='Buzz'\nassert r[14]=='FizzBuzz'\nprint('PASS')"),
    ("Write `binary_search(arr, target)` returning the index or -1.",
     "Two-pointer binary search on sorted array.",
     "def binary_search(arr, t):\n    lo,hi=0,len(arr)-1\n    while lo<=hi:\n        m=(lo+hi)//2\n        if arr[m]==t: return m\n        elif arr[m]<t: lo=m+1\n        else: hi=m-1\n    return -1",
     "assert binary_search([1,3,5,7,9],5)==2\nassert binary_search([1,3,5,7,9],6)==-1\nprint('PASS')"),
    ("Write `count_words(text)` returning a word-frequency dict.",
     "Split on whitespace and count each word.",
     "def count_words(text):\n    c={}\n    for w in text.lower().split(): c[w]=c.get(w,0)+1\n    return c",
     "r=count_words('hello world hello')\nassert r['hello']==2\nassert r['world']==1\nprint('PASS')"),
    ("Write `merge_sorted(a, b)` merging two sorted lists.",
     "Two-pointer merge into a single sorted list.",
     "def merge_sorted(a,b):\n    r,i,j=[],0,0\n    while i<len(a) and j<len(b):\n        if a[i]<=b[j]: r.append(a[i]);i+=1\n        else: r.append(b[j]);j+=1\n    return r+a[i:]+b[j:]",
     "assert merge_sorted([1,3],[2,4])==[1,2,3,4]\nprint('PASS')"),
    ("Write `max_subarray(nums)` using Kadane's algorithm.",
     "Track current and max sum; reset current if it goes negative.",
     "def max_subarray(nums):\n    mx=cur=nums[0]\n    for n in nums[1:]:\n        cur=max(n,cur+n); mx=max(mx,cur)\n    return mx",
     "assert max_subarray([-2,1,-3,4,-1,2,1,-5,4])==6\nprint('PASS')"),
    ("Write `two_sum(nums, target)` returning two indices that sum to target.",
     "Use a hash map: store each num's index, check complement.",
     "def two_sum(nums,target):\n    seen={}\n    for i,n in enumerate(nums):\n        if target-n in seen: return [seen[target-n],i]\n        seen[n]=i",
     "assert two_sum([2,7,11,15],9)==[0,1]\nprint('PASS')"),
    ("Write `flatten(lst)` that flattens a list one level deep.",
     "Extend result with each sublist, append scalars directly.",
     "def flatten(lst):\n    r=[]\n    for item in lst:\n        if isinstance(item,list): r.extend(item)\n        else: r.append(item)\n    return r",
     "assert flatten([[1,2],[3],[4,5]])==[1,2,3,4,5]\nprint('PASS')"),
    ("Write `longest_common_prefix(strs)` returning the longest common prefix.",
     "Zip all strings and compare character sets.",
     "def longest_common_prefix(strs):\n    if not strs: return ''\n    pref=''\n    for chars in zip(*strs):\n        if len(set(chars))>1: break\n        pref+=chars[0]\n    return pref",
     "assert longest_common_prefix(['flower','flow','flight'])=='fl'\nassert longest_common_prefix(['dog','racecar','car'])==''\nprint('PASS')"),
    ("Write `is_valid_parens(s)` returning True if all brackets are balanced.",
     "Use a stack; pop when closing bracket matches top.",
     "def is_valid_parens(s):\n    stack=[]; m={')':'(',']':'[','}':'{'}\n    for c in s:\n        if c in '([{': stack.append(c)\n        elif not stack or stack[-1]!=m[c]: return False\n        else: stack.pop()\n    return not stack",
     "assert is_valid_parens('()[]{}')\nassert not is_valid_parens('(]')\nprint('PASS')"),
    ("Write `power(base, exp)` without using ** operator.",
     "Repeated multiplication in a loop.",
     "def power(base,exp):\n    r=1\n    for _ in range(exp): r*=base\n    return r",
     "assert power(2,10)==1024\nassert power(3,0)==1\nprint('PASS')"),
    ("Write `roman_to_int(s)` converting a Roman numeral to an integer.",
     "Walk left to right; subtract if current value is less than next.",
     "def roman_to_int(s):\n    m={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n    r=0\n    for i,c in enumerate(s):\n        if i+1<len(s) and m[c]<m[s[i+1]]: r-=m[c]\n        else: r+=m[c]\n    return r",
     "assert roman_to_int('III')==3\nassert roman_to_int('IV')==4\nassert roman_to_int('IX')==9\nprint('PASS')"),
    ("Write `climbing_stairs(n)` counting ways to climb n stairs (1 or 2 steps).",
     "Fibonacci-style DP: ways(n) = ways(n-1) + ways(n-2).",
     "def climbing_stairs(n):\n    a,b=1,1\n    for _ in range(n-1): a,b=b,a+b\n    return b",
     "assert climbing_stairs(1)==1\nassert climbing_stairs(2)==2\nassert climbing_stairs(5)==8\nprint('PASS')"),
    ("Write `is_prime(n)` returning True if n is a prime number.",
     "Trial division up to sqrt(n); return False if any divisor found.",
     "def is_prime(n):\n    if n<2: return False\n    for i in range(2,int(n**0.5)+1):\n        if n%i==0: return False\n    return True",
     "assert is_prime(2) and is_prime(17) and not is_prime(1) and not is_prime(4)\nprint('PASS')"),
    ("Write `rotate_list(lst, k)` rotating a list right by k positions.",
     "Slice with modulo to avoid unnecessary full rotations.",
     "def rotate_list(lst,k):\n    if not lst: return lst\n    k%=len(lst); return (lst[-k:]+lst[:-k]) if k else lst[:]",
     "assert rotate_list([1,2,3,4,5],2)==[4,5,1,2,3]\nassert rotate_list([1,2,3],0)==[1,2,3]\nprint('PASS')"),
    ("Write `sum_digits(n)` returning the sum of digits of n.",
     "Convert to string and sum ord values, or use modulo arithmetic.",
     "def sum_digits(n):\n    return sum(int(d) for d in str(abs(n)))",
     "assert sum_digits(123)==6\nassert sum_digits(0)==0\nassert sum_digits(999)==27\nprint('PASS')"),
    ("Write `find_duplicates(lst)` returning a list of elements that appear more than once.",
     "Count occurrences, filter those > 1.",
     "def find_duplicates(lst):\n    counts={}\n    for x in lst: counts[x]=counts.get(x,0)+1\n    return [x for x,c in counts.items() if c>1]",
     "assert sorted(find_duplicates([1,2,2,3,3,3]))==[2,3]\nassert find_duplicates([1,2,3])==[]\nprint('PASS')"),
    ("Write `intersection(a, b)` returning elements common to both lists (no duplicates).",
     "Use set intersection.",
     "def intersection(a,b):\n    return list(set(a)&set(b))",
     "assert sorted(intersection([1,2,3,4],[2,4,6]))==[2,4]\nassert intersection([1,2],[3,4])==[]\nprint('PASS')"),
]

out = Path(__file__).parent.parent / "data" / "local_test.jsonl"
with open(out, "w") as f:
    for prompt, thinking, solution, test_code in EXAMPLES:
        # Format A — must match scripts/prepare_sft_data.py: no <think>, and the
        # assistant turn ends with the code (no closing ``` fence). A closing
        # fence teaches the model ``` -> <|im_end|>, collapsing generation.
        assistant = f"```python\n{solution}\n"
        chatml = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>"
        )
        f.write(json.dumps({
            "source": "local_test",
            "chatml": chatml,
            "prompt": prompt,
            "test_code": test_code,
            "language": "python",
        }) + "\n")

print(f"[DATA] Written {len(EXAMPLES)} coding examples → {out}")
