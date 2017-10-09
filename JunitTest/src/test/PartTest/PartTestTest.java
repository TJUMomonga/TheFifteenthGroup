package PartTest;


import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by ltp on 2017/10/9.
 */
public class PartTestTest extends PartTest {
    @Before
    public void setUp() throws Exception {

    }

    @After
    public void tearDown() throws Exception {

    }

    @Test
    public void main() throws Exception {
        assertEquals("ltp",new PartTest());
    }

}